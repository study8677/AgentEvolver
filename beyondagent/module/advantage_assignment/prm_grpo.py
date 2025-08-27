# -*- coding: utf-8 -*-
# PRM step → (optional) group-level standardization on steps → per-trajectory projection (optional) → suffix-sum on steps → broadcast to tokens
from __future__ import annotations
from typing import List, Dict, Optional
from dataclasses import dataclass
import torch
import math

# =========================
# Hyper & small utilities
# =========================

@dataclass
class PRMHyper:
    # 权重：一致性步的权重大，不一致性步的权重小（用于 allocation / allocation_c）
    consistent_scale: float = 1.0
    pos_unconsistent_scale: float = 0.2   # 成功轨迹里的 BAD 步权重
    neg_unconsistent_scale: float = 0.2   # 失败轨迹里的 GOOD 步权重
    eps: float = 1e-8
    do_batch_zscore: bool = True          # 是否做组内 z-score（按 step 级，allocation_c/decouple 会用到）
    traj_equal_zscore: bool = True        # True=每条轨迹等权；False=把所有 step 拉平成一个大样本
    fix_base: float = 0.2                 # fix 方案的基础幅度（good=+base, bad=-base）
    alpha: float = 1.0                    # PRM权重平衡系数
    orm_distribution: str = "last_step"   # ORM分配方式："last_step" 或 "all_steps"

def _ensure_tensor(x, device, dtype=None):
    if torch.is_tensor(x):
        t = x.to(device=device)
        if dtype is not None:
            t = t.to(dtype)
        return t
    return torch.as_tensor(x, device=device, dtype=dtype)

def _num_steps_from_step_ids(step_ids_row: torch.Tensor) -> int:
    """step_ids: shape (L,) with -1 for non-response tokens; contiguous step ids starting at 0."""
    if step_ids_row.numel() == 0:
        return 0
    m = torch.amax(step_ids_row)
    return int(m.item() + 1) if m.item() >= 0 else 0

def _align_flags(flags: List[bool], K: int, is_success: bool) -> List[bool]:
    if len(flags) == K:
        return list(flags)
    default_flag = True if is_success else False
    if len(flags) < K:
        return list(flags) + [default_flag] * (K - len(flags))
    else:
        return list(flags[:K])

# =========================
# Z-score helpers (group-wise, step-level)
# =========================

def _group_zscore_on_steps(
    step_rewards_raw: List[List[float]],
    group_ids: torch.Tensor,
    hyper: PRMHyper,
) -> List[List[float]]:
    """对 step 奖励做“组内”减均值/除方差标准化。
    - traj_equal_zscore=True: 每条轨迹等权；组均值 = 轨迹均值的均值；
      组方差 = 轨迹内相对组均值的均方差的均值（second-moment around group mean）
    - traj_equal_zscore=False: 拉平本组所有 step 一起算
    """
    if not hyper.do_batch_zscore:
        return [list(r) for r in step_rewards_raw]

    B = len(step_rewards_raw)
    gids = group_ids.view(-1).tolist()
    g2idx: Dict[int, List[int]] = {}
    for i, g in enumerate(gids):
        g2idx.setdefault(int(g), []).append(i)

    step_rewards_std: List[List[float]] = [[] for _ in range(B)]
    for _, idxs in g2idx.items():
        if hyper.traj_equal_zscore:
            # 1) 组均值：轨迹均值的等权平均
            traj_means = []
            for i in idxs:
                ri = step_rewards_raw[i]
                if ri:
                    traj_means.append(sum(ri) / len(ri))
            if len(traj_means) == 0:
                mu_g, sd_g = 0.0, 1.0
            else:
                mu_g = float(sum(traj_means) / len(traj_means))
                # 2) 组方差：先对每条轨迹围绕 mu_g 求均方差，再对轨迹等权平均
                second_moments = []
                for i in idxs:
                    ri = step_rewards_raw[i]
                    if not ri:
                        continue
                    second_moments.append(sum((x - mu_g) * (x - mu_g) for x in ri) / len(ri))
                var_g = float(sum(second_moments) / len(second_moments)) if second_moments else 0.0
                sd_g = float(math.sqrt(var_g + hyper.eps))
        else:
            # 拉平：把本组所有 step 拼在一起
            flat = []
            for i in idxs:
                flat.extend(step_rewards_raw[i])
            if len(flat) == 0:
                mu_g, sd_g = 0.0, 1.0
            else:
                t = torch.tensor(flat, dtype=torch.float32)
                mu_g = float(t.mean().item())
                sd_g = float(max(t.std(unbiased=False).item(), hyper.eps))

        inv = 1.0 / (sd_g + 1e-12)
        for i in idxs:
            ri = step_rewards_raw[i]
            if not ri:
                step_rewards_std[i] = []
            else:
                step_rewards_std[i] = [float((x - mu_g) * inv) for x in ri]
    return step_rewards_std

def _per_traj_scale_to_target_sum(
    r_std: List[float],
    target_sum: float,
    eps: float,
) -> List[float]:
    """把一条轨迹的 step 列表按比例缩放，使其总和=target_sum。退化时均分。"""
    if len(r_std) == 0:
        return []
    cur = sum(r_std)
    if abs(cur) <= eps:
        return [target_sum / len(r_std) for _ in r_std]
    scale = target_sum / cur
    return [float(x * scale) for x in r_std]

# =========================
# Builders for 4 schemes
# =========================

def _build_fix(
    orms_sign: torch.Tensor,
    step_flags: List[List[bool]],
    step_ids: torch.Tensor,
    hyper: PRMHyper,
) -> List[List[float]]:
    """方案1：fix —— 固定基数（±base），不强制 ∑=±1。
    成功/失败仅通过 orms_sign 决定整体方向：r_step = sign(ORM) * ( +base if good else -base )
    """
    B = step_ids.size(0)
    out: List[List[float]] = []
    base = float(hyper.fix_base)
    for i in range(B):
        K = _num_steps_from_step_ids(step_ids[i])
        if K == 0:
            out.append([]); continue
        sgn = 1.0 if float(orms_sign[i].item()) > 0 else -1.0
        # 对齐 flags；默认填充 good=True 只是保证长度，这里整体方向由 sgn 控制
        flags = _align_flags(step_flags[i] if i < len(step_flags) else [], K, is_success=True)
        r = [sgn * (+base if f else -base) for f in flags]
        out.append([float(x) for x in r])
    return out

def _build_allocation(
    orms_sign: torch.Tensor,
    step_flags: List[List[bool]],
    step_ids: torch.Tensor,
    hyper: PRMHyper,
) -> List[List[float]]:
    """方案2：allocation —— 一致性瓜分（同号），不做标准化；逐轨迹 ∑=±1。"""
    B = step_ids.size(0)
    out: List[List[float]] = []
    for i in range(B):
        K = _num_steps_from_step_ids(step_ids[i])
        if K == 0:
            out.append([]); continue
        is_success = bool(orms_sign[i].item() > 0)
        flags = _align_flags(step_flags[i] if i < len(step_flags) else [], K, is_success)
        n_g = sum(1 for f in flags if f); n_b = K - n_g
        if is_success:
            w_g, w_b = hyper.consistent_scale, hyper.pos_unconsistent_scale
            sgn = +1.0
        else:
            w_g, w_b = hyper.neg_unconsistent_scale, hyper.consistent_scale
            sgn = -1.0
        total_w = n_g * w_g + n_b * w_b
        unit = 0.0 if total_w <= hyper.eps else (1.0 / total_w)
        # 同号瓜分：good 和 bad 都随 orms_sign 同号，仅幅度不同；确保 sum = ±1
        r = [sgn * (w_g * unit) if f else sgn * (w_b * unit) for f in flags]
        out.append([float(x) for x in r])
    return out

def _build_allocation_c(
    orms_sign: torch.Tensor,
    step_flags: List[List[bool]],
    step_ids: torch.Tensor,
    group_ids: torch.Tensor,
    hyper: PRMHyper,
) -> List[List[float]]:
    """方案3：allocation_c —— 一致性瓜分（同号） → 组内 z-score → 按比例缩放投影（∑=±1）。"""
    B = step_ids.size(0)
    # 1) raw（逐轨迹 ∑=±1）
    step_rewards_raw: List[List[float]] = []
    for i in range(B):
        K = _num_steps_from_step_ids(step_ids[i])
        if K == 0:
            step_rewards_raw.append([]); continue
        is_success = bool(orms_sign[i].item() > 0)
        flags = _align_flags(step_flags[i] if i < len(step_flags) else [], K, is_success)
        n_g = sum(1 for f in flags if f); n_b = K - n_g
        if is_success:
            w_g, w_b = hyper.consistent_scale, hyper.pos_unconsistent_scale
            sgn = +1.0
        else:
            w_g, w_b = hyper.neg_unconsistent_scale, hyper.consistent_scale
            sgn = -1.0
        total_w = n_g * w_g + n_b * w_b
        unit = 0.0 if total_w <= hyper.eps else (1.0 / total_w)
        r_raw = [sgn * (w_g * unit) if f else sgn * (w_b * unit) for f in flags]
        step_rewards_raw.append([float(x) for x in r_raw])
    # 2) group z-score
    r_std = _group_zscore_on_steps(step_rewards_raw, group_ids, hyper)
    # 3) 按比例缩放投影（逐轨迹 ∑=±1）
    out: List[List[float]] = []
    for i in range(B):
        out.append(_per_traj_scale_to_target_sum(r_std[i], float(orms_sign[i].item()), eps=hyper.eps))
    return out

def _build_decouple(
    orm_full_scores: torch.Tensor,  # 完整的ORM分数
    step_flags: List[List[bool]],
    step_ids: torch.Tensor,
    group_ids: torch.Tensor,
    hyper: PRMHyper
) -> List[List[float]]:
    """方案4：decouple —— PRM 和 ORM 分别标准化后组合；不强制 ∑=±1。
    - PRM：基于 flags 构造基础奖励，做组内 z-score 标准化
    - ORM：使用完整的 ORM 分数，做组内 z-score 标准化
    - 组合：alpha * normalized_prm + normalized_orm（按 orm_distribution 方式分配）
    """
    B = step_ids.size(0)
    device = step_ids.device
    alpha = hyper.alpha
    orm_distribution = hyper.orm_distribution
    
    # ---- 1. 构造基础 PRM 奖励（与 ORM 无关）----
    prm_rewards_raw: List[List[float]] = []
    for i in range(B):
        K = _num_steps_from_step_ids(step_ids[i])
        if K == 0:
            prm_rewards_raw.append([])
            continue
        flags = _align_flags(step_flags[i] if i < len(step_flags) else [], K, is_success=True)
        # 使用固定的 good/bad 值，不依赖 ORM
        prm_rewards = [hyper.fix_base if f else -hyper.fix_base for f in flags]
        prm_rewards_raw.append(prm_rewards)
    
    # ---- 2. 对 PRM 奖励做组内 z-score 标准化 ----
    prm_rewards_std = _group_zscore_on_steps(prm_rewards_raw, group_ids, hyper)
    
    # ---- 3. 对完整的 ORM 分数做组内标准化 ----
    orm_scores = orm_full_scores.cpu().tolist()
    
    # 对 ORM 分数做组内标准化
    gids = group_ids.view(-1).tolist()
    g2idx: Dict[int, List[int]] = {}
    for i, g in enumerate(gids):
        g2idx.setdefault(int(g), []).append(i)
    
    orm_scores_std = [0.0] * B
    for _, idxs in g2idx.items():
        group_orms = [orm_scores[i] for i in idxs]
        if len(group_orms) == 0:
            continue
        orm_tensor = torch.tensor(group_orms, dtype=torch.float32)
        orm_mean = orm_tensor.mean()
        orm_std = orm_tensor.std(unbiased=False)
        
        if orm_std <= hyper.eps:
            # 标准差太小，只减均值
            for i in idxs:
                orm_scores_std[i] = float(orm_scores[i] - orm_mean.item())
        else:
            # 标准 z-score
            for i in idxs:
                orm_scores_std[i] = float((orm_scores[i] - orm_mean.item()) / (orm_std.item() + 1e-12))
    
    # ---- 4. 组合标准化的 PRM 和 ORM ----
    combined_rewards: List[List[float]] = []
    for i in range(B):
        if not prm_rewards_std[i]:
            combined_rewards.append([])
            continue
        
        prm_std = prm_rewards_std[i]
        orm_std = orm_scores_std[i]
        
        combined = []
        for j, prm_reward in enumerate(prm_std):
            if orm_distribution == "last_step":
                # ORM 只加在最后一步
                if j == len(prm_std) - 1:
                    combined_reward = alpha * prm_reward + orm_std
                else:
                    combined_reward = alpha * prm_reward
            elif orm_distribution == "all_steps":
                # ORM 每步都加
                combined_reward = alpha * prm_reward + orm_std
            else:
                raise ValueError(f"Unknown orm_distribution: {orm_distribution}")
            
            combined.append(float(combined_reward))
        
        combined_rewards.append(combined)
    
    return combined_rewards

# =========================
# Step → Token broadcast + suffix-sum
# =========================

def suffix_sum_on_steps(step_rewards: List[List[float]]) -> List[List[float]]:
    """对每个样本的 step 回报做后缀和，输出同形状的 step-adv。"""
    adv: List[List[float]] = []
    for r in step_rewards:
        if not r:
            adv.append([]); continue
        t = torch.tensor(r, dtype=torch.float32)
        s = torch.flip(torch.cumsum(torch.flip(t, dims=[0]), dim=0), dims=[0])
        adv.append([float(x) for x in s])
    return adv

def broadcast_step_adv_to_tokens(
    step_adv: List[List[float]],
    step_ids: torch.Tensor,
) -> torch.Tensor:
    """把 step-adv 按 step_ids 广播到 token 上。step_ids 为 -1 的位置填 0。"""
    device = step_ids.device
    B, L = step_ids.shape
    out = torch.zeros((B, L), device=device, dtype=torch.float32)
    for i in range(B):
        if not step_adv[i]:
            continue
        adv_i = torch.tensor(step_adv[i], device=device, dtype=torch.float32)
        sid_row = step_ids[i]
        valid = sid_row >= 0
        if torch.any(valid):
            sids = sid_row[valid]
            out[i, valid] = adv_i[sids]
    return out

# =========================
# Entry
# =========================

def compute_prm_grpo_advantages(
    batch,                          # DataProto 或兼容结构：batch.batch[...] 可索引
    step_flags: List[List[bool]],   # 每条轨迹的 GOOD/BAD 标志
    hyper: Optional[PRMHyper] = None,
    scheme: str = "allocation_c",   # "fix" | "allocation" | "allocation_c" | "decouple"
) -> dict:
    """
    统一入口：
      - 先把 ORM 压成 ±1：orms_sign = sign(sum(token_level_rewards)) （== +1 if sum>0 else -1）
      - 根据 scheme 构造 step-level 奖励（见各 builder），得到 step_rewards
      - step 后缀和 → step_adv
      - 广播到 token → advantages (B, L)
    返回：
      - advantages: (B, L) token-level advantages
      - orm_scalar: (B,) 逐条轨迹的 ±1
    """
    if hyper is None:
        hyper = PRMHyper()

    # ---- 取必要字段 ----
    responses = batch.batch["responses"]
    device = responses.device if torch.is_tensor(responses) else torch.as_tensor(responses).device

    step_ids = _ensure_tensor(batch.batch["step_ids"], device=device, dtype=torch.long)      # (B, L_resp) with -1 for non-response
    group_ids = _ensure_tensor(batch.batch["group_ids"], device=device, dtype=torch.long).view(-1)

    # 取 token-level reward（可能字段名不同，做兜底）
    token_keys_try = ["token_level_rewards", "response_token_level_rewards", "token_rewards"]
    token_level_rewards = None
    for k in token_keys_try:
        if k in batch.batch:
            token_level_rewards = _ensure_tensor(batch.batch[k], device=device, dtype=torch.float32)
            break
    if token_level_rewards is None:
        raise KeyError("token-level rewards not found in batch (tried keys: token_level_rewards / response_token_level_rewards / token_rewards)")

    # ---- ORM_sign = ±1（保持 sum>0 → +1；sum<=0 → -1）----
    # TODO: ORM做normalization
    orm_sum = token_level_rewards.sum(dim=1)   # (B,)
    orms_score = torch.where(orm_sum > 0, torch.ones_like(orm_sum), -torch.ones_like(orm_sum)).to(dtype=torch.float32)

    # ---- Build step rewards by scheme ----
    scheme = (scheme or "allocation_c").lower()
    if scheme == "fix":
        step_rewards = _build_fix(orms_score, step_flags, step_ids, hyper)
    elif scheme == "allocation":
        step_rewards = _build_allocation(orms_score, step_flags, step_ids, hyper)
    elif scheme == "allocation_c":
        step_rewards = _build_allocation_c(orms_score, step_flags, step_ids, group_ids, hyper)
    elif scheme == "decouple":
        step_rewards = _build_decouple(orms_score, step_flags, step_ids, group_ids, hyper,)
    else:
        raise ValueError(f"Unknown PRM scheme: {scheme} (expected one of: fix | allocation | allocation_c | decouple)")

    # ---- Step → token advantages ----
    step_adv = suffix_sum_on_steps(step_rewards)
    advantages = broadcast_step_adv_to_tokens(step_adv, step_ids)

    return {
        "advantages": advantages,        # (B, L_resp)
        "orm_scalar": orms_sign,         # (B,)
    }

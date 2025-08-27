from typing import List, Sequence
import torch
from dataclasses import dataclass
from beyondagent.module.advantage_assignment.test_allocation import *
@dataclass
class PRMHyper:
    consistent_scale: float = 1.0
    pos_unconsistent_scale: float = 0.2  # 成功轨迹的 BAD 步骤权重
    neg_unconsistent_scale: float = 0.2  # 失败轨迹的 GOOD 步骤权重
    eps: float = 1e-8
    do_batch_zscore: bool = True         # 是否做批内 z-score（减均值/除方差）

def compute_step_rewards_from_flags_consistent_centered(
    orms: List[float],
    step_flags_list: List[List[bool]],
    hyper: PRMHyper,
) -> List[List[float]]:
    """
    一致性瓜分 + 批内减均值标准化 + 轨迹内再投影（保持 ∑step = ORM）
    步骤：
      1) 一致性权重瓜分得到 r_raw（每条轨迹和=ORM）
      2) 批内 z-score：r_std = (r_raw - mean) / std   （控幅；可能改变符号）
      3) 逐轨迹去均值 + 加回 ORM/S：r_proj = (r_std - mean_traj) + ORM/S  （恢复每轨迹和=ORM）
    """
    step_rewards_raw: List[List[float]] = []

    # --- 1) 一致性瓜分 ---
    for i, (orm, flags) in enumerate(zip(orms, step_flags_list)):
        S = len(flags)
        if S == 0:
            step_rewards_raw.append([])
            continue

        is_success = (orm >= 0.0)
        if is_success:
            w_good = hyper.consistent_scale
            w_bad  = hyper.pos_unconsistent_scale
        else:
            w_good = hyper.neg_unconsistent_scale
            w_bad  = hyper.consistent_scale

        weights = [(w_good if f else w_bad) for f in flags]
        total_w = sum(weights)
        unit = 0.0 if abs(total_w) <= hyper.eps else (orm / total_w)
        r_raw = [w * unit for w in weights]
        step_rewards_raw.append(r_raw)

        # 校验瓜分和
        s = sum(r_raw)
        print(f"[PRM-RAW] 样本{i}: ORM={orm:.3f}, ∑raw={s:.6f}, diff={abs(orm - s):.8f}")
    print("r_raw:", step_rewards_raw)
    import pdb;pdb.set_trace()
    # 若不做 z-score，直接返回 raw
    if not hyper.do_batch_zscore:
        return step_rewards_raw

    # --- 2) 批内 z-score（减均值/除方差） ---
    # 将所有 r_raw 拉平做批统计
    flat = [x for lst in step_rewards_raw for x in lst]
    
    if len(flat) == 0:
        return step_rewards_raw

    t = torch.tensor(flat, dtype=torch.float32)
    mean = t.mean()
    std  = t.std(unbiased=False)

    # 避免除零：std 太小则只做减均值
    def _z(x: float) -> float:
        if std <= hyper.eps:
            return float(x - mean.item())
        else:
            return float((x - mean.item()) / (std.item() + 1e-12))

    step_rewards_std: List[List[float]] = [[_z(x) for x in r_raw] for r_raw in step_rewards_raw]
    print("step_rewards_raw:", step_rewards_raw)
    import pdb;pdb.set_trace()
    # --- 3) 逐轨迹去均值 + ORM/S 再投影（保证每条轨迹和=ORM） ---
    step_rewards_proj: List[List[float]] = []
    for orm, r_std in zip(orms, step_rewards_std):
        if len(r_std) == 0:
            step_rewards_proj.append([])
            continue
        traj_mean = sum(r_std) / len(r_std)
        # 去掉本轨迹均值，只改变平移，不改相对形状
        r_centered = [x - traj_mean for x in r_std]
        # 加回 ORM/S（恢复“和=ORM”的约束）
        r_proj = [x + orm / len(r_centered) for x in r_centered]
        step_rewards_proj.append(r_proj)
    print("step_rewards_proj",step_rewards_proj)
    import pdb;pdb.set_trace()
    # 再次校验和
    for i, (orm, rr) in enumerate(zip(orms, step_rewards_proj)):
        s = sum(rr)
        print(f"[PRM-PROJ] 样本{i}: ORM={orm:.3f}, ∑proj={s:.6f}, diff={abs(orm - s):.8f}")

    return step_rewards_proj

def grpo_advantage_process_steps_centered(
    step_rewards_list: Sequence[Sequence[float]],
) -> List[List[float]]:
    """
    GRPO 优势：对已“减均值标准化+再投影”的 step rewards 只做后缀和。
    """
    advantages_list: List[List[float]] = []
    for rewards in step_rewards_list:
        t = torch.as_tensor(rewards, dtype=torch.float32)
        adv = torch.flip(torch.cumsum(torch.flip(t, dims=[0]), dim=0), dims=[0]).tolist()
        advantages_list.append(adv)
    return advantages_list

def test_prm_grpo_consistent():
    print("=" * 60)
    print("一致性瓜分 PRM + 无减均值 GRPO 后缀和（保号）测试")
    print("=" * 60)
    
    hyper = PRMHyper()

    # # 测试1: 成功轨迹批次，多数good步骤
    # print("\n【测试1: 成功轨迹批次，多数GOOD步骤】")
    # orms1 = [1.0, 1.0]
    # flags1 = [
    #     [True, True, False, True, True, False, True, True, True],  # 7G,2B
    #     [True, True, True, False, True, True, False]               # 5G,2B
    # ]
    # step_rewards1 = compute_step_rewards_from_flags_consistent_centered(orms1, flags1, hyper)
    # advantages1 = grpo_advantage_process_steps_centered(step_rewards1)
    # for i in range(len(orms1)):
    #     print(f"样本{i} (ORM={orms1[i]}):")
    #     print(f"  Flags: {flags1[i]}")
    #     print(f"  Rewards: {[f'{x:.4f}' for x in step_rewards1[i]]}")
    #     print(f"  Advantages: {[f'{x:.4f}' for x in advantages1[i]]}")
    #     check_order_per_sample(flags1[i], step_rewards1[i], orms1[i])

    # # 测试2: 失败轨迹批次，多数bad步骤
    # print("\n【测试2: 失败轨迹批次，多数BAD步骤】")
    # orms2 = [-1.0, -1.0]
    # flags2 = [
    #     [True, False, False, True, False, False, False, False, False],  # 2G,7B
    #     [False, True, False, False, False, False]                       # 1G,5B
    # ]
    # step_rewards2 = compute_step_rewards_from_flags_consistent_centered(orms2, flags2, hyper)
    # advantages2 = grpo_advantage_process_steps_centered(step_rewards2)
    # for i in range(len(orms2)):
    #     print(f"样本{i} (ORM={orms2[i]}):")
    #     print(f"  Flags: {flags2[i]}")
    #     print(f"  Rewards: {[f'{x:.4f}' for x in step_rewards2[i]]}")
    #     print(f"  Advantages: {[f'{x:.4f}' for x in advantages2[i]]}")
    #     check_order_per_sample(flags2[i], step_rewards2[i], orms2[i])

    # # 测试3: 混合批次
    # print("\n【测试3: 混合批次】")
    # orms3 = [1.0, -1.0]
    # flags3 = [
    #     [True, True, False, True, True],      # 成功: 4G,1B
    #     [False, False, True, False, False]    # 失败: 1G,4B
    # ]
    # step_rewards3 = compute_step_rewards_from_flags_consistent_centered(orms3, flags3, hyper)
    # advantages3 = grpo_advantage_process_steps_centered(step_rewards3)
    # for i in range(len(orms3)):
    #     print(f"\n样本{i} (ORM={orms3[i]}):")
    #     print(f"  Flags: {flags3[i]}")
    #     print(f"  Rewards: {[f'{x:.4f}' for x in step_rewards3[i]]}")
    #     print(f"  Advantages: {[f'{x:.4f}' for x in advantages3[i]]}")
    #     check_order_per_sample(flags3[i], step_rewards3[i], orms3[i])

    # # 测试4: 边界情况
    # print("\n【测试4: 边界情况】")
    # print("4a. 全GOOD：")
    # orms4a = [1.0, 1.0]
    # flags4a = [[True, True, True], [True, True, True, True]]
    # step_rewards4a = compute_step_rewards_from_flags_consistent_centered(orms4a, flags4a, hyper)
    # advantages4a = grpo_advantage_process_steps_centered(step_rewards4a)
    # for i in range(len(orms4a)):
    #     print(f"  样本{i} (ORM={orms4a[i]}):")
    #     print(f"    Rewards: {[f'{x:.4f}' for x in step_rewards4a[i]]}")
    #     print(f"    Advantages: {[f'{x:.4f}' for x in advantages4a[i]]}")
    #     check_order_per_sample(flags4a[i], step_rewards4a[i], orms4a[i])

    # print("4b. 全BAD：")
    # orms4b = [-1.0, -1.0]
    # flags4b = [[False, False, False], [False, False, False, False]]
    # step_rewards4b = compute_step_rewards_from_flags_consistent_centered(orms4b, flags4b, hyper)
    # advantages4b = grpo_advantage_process_steps_centered(step_rewards4b)
    # for i in range(len(orms4b)):
    #     print(f"  样本{i} (ORM={orms4b[i]}):")
    #     print(f"    Rewards: {[f'{x:.4f}' for x in step_rewards4b[i]]}")
    #     print(f"    Advantages: {[f'{x:.4f}' for x in advantages4b[i]]}")
    #     check_order_per_sample(flags4b[i], step_rewards4b[i], orms4b[i])

    print("特别情况：")
    orms = [1.0, -1.0, 1.0, 1.0]
    flags = [
        [True, True, False, True, True],      # 成功: 4G,1B
        [False, False, True, False, False] ,   # 失败: 1G,4B,      # 成功: 4G,1B
        [True, True, False, True, True, False, True, True, True],  # 7G,2B
        [True, True, True, False, True]    # 成功: 1G,4B
    ]
    step_rewards = compute_step_rewards_from_flags_consistent_centered(orms, flags, hyper)
    advantages = grpo_advantage_process_steps_centered(step_rewards)
    for i in range(len(orms)):
        print(f"  样本{i} (ORM={orms[i]}):")
        print(f"    Rewards: {[f'{x:.4f}' for x in step_rewards[i]]}")
        print(f"    Advantages: {[f'{x:.4f}' for x in advantages[i]]}")
        check_order_per_sample(flags[i], step_rewards[i], orms[i])

if __name__ == "__main__":
    test_prm_grpo_consistent()

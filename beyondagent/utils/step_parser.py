# shuchang: 0809
# FIXME: 这个文件是step_parser.py，功能：把解析模型的response_id解析为step，统一所有需要step的模块
from dataclasses import dataclass
from typing import List, Dict, Tuple
import torch

@dataclass
class StepParseResult:
    segments: List[Dict]         # [{'role': str, 'start': int, 'end': int, 'tokens': List[int]}]
    steps: List[Dict]            # [{'action_tokens': List[int], 'observation_tokens': List[int],
                                #   'action_text': str, 'observation_text': str, 
                                #   'action_start': int, 'action_end': int, 'obs_start': int, 'obs_end': int}]
    step_ids: List[int]          # len == len(response_ids); assistant动作区间标k，其余-1

def _find_first_subseq(hay, needle):
    """安全的子序列搜索，避免单token误匹配"""
    if not needle:
        return None
    L = len(needle)
    for i in range(len(hay) - L + 1):
        if hay[i:i+L] == needle:
            return i
    return None

def _locate_template_positions(tokens: List[int], tpl: List[int]) -> List[int]:
    """返回 tpl 在 tokens 中出现的起点索引"""
    if not tpl:  # 保护：避免空模板死循环
        return []
    
    pos, out, L = 0, [], len(tpl)
    while pos <= len(tokens) - L:
        if tokens[pos:pos+L] == tpl:
            out.append(pos)
            pos += L
        else:
            pos += 1
    return out

def _extract_role_header_tokens(tokenizer, role: str) -> List[int]:
    """
    通用方法：自动提取任何模型的role header tokens
    原理：通过对比空内容和带内容的消息，找出role header部分
    如果提取失败，直接抛出异常
    """
    try:
        if role == "assistant":
            # 比较不带assistant回复 vs 带assistant回复的差异
            user_only = [{"role": "user", "content": ""}]
            user_tokens = tokenizer.apply_chat_template(
                user_only, tokenize=True, add_generation_prompt=False
            )
            
            # 带assistant的完整对话
            full_dialog = [{"role": "user", "content": ""}, {"role": "assistant", "content": "x"}]
            full_tokens = tokenizer.apply_chat_template(
                full_dialog, tokenize=True, add_generation_prompt=False
            )
            
            # 找到"x"的位置（使用安全的子序列搜索）
            x_tokens = tokenizer.encode("x", add_special_tokens=False)
            if not x_tokens:
                raise ValueError(f"Cannot encode 'x' token for role {role}")
            
            
            x_position = _find_first_subseq(full_tokens, x_tokens)
            if x_position is None:
                raise ValueError(f"Cannot find 'x' token sequence in full dialog for role {role}")
            
            
            # assistant header = 从user_only结束到"x"开始的部分
            user_len = len(user_tokens)
            
            if user_len < x_position:
                header_tokens = full_tokens[user_len:x_position]
                return header_tokens
            elif user_len == x_position:
                return []  # 返回空header，这是合法情况
            else:
                raise ValueError(f"Invalid token positions for role {role}: user_len={user_len}, x_pos={x_position}")
                
        else:
            # 对于user等其他角色：比较空内容vs带内容
            # 关键修复：不要让user模板包含system message
            empty_msg = [{"role": role, "content": ""}]
            empty_tokens = tokenizer.apply_chat_template(
                empty_msg, tokenize=True, add_generation_prompt=False
            )
            
            content_msg = [{"role": role, "content": "x"}]
            content_tokens = tokenizer.apply_chat_template(
                content_msg, tokenize=True, add_generation_prompt=False
            )
            
            # 找到"x"的位置（使用安全的子序列搜索）
            x_tokens = tokenizer.encode("x", add_special_tokens=False)
            if not x_tokens:
                raise ValueError(f"Cannot encode 'x' token for role {role}")
            
            x_position = _find_first_subseq(content_tokens, x_tokens)
            if x_position is None:
                raise ValueError(f"Cannot find 'x' token sequence in content message for role {role}")
            
            # 关键修复：用更精确的方法提取纯role header
            # 对于content_msg，x前面的部分应该是role header
            # 但如果empty_tokens包含了额外内容（如system），需要排除
            
            if len(content_tokens) > len(empty_tokens):
                # 新增部分是 header + "x"
                added_part = content_tokens[len(empty_tokens):]
                x_pos_in_added = _find_first_subseq(added_part, x_tokens)
                if x_pos_in_added is not None:
                    header_tokens = added_part[:x_pos_in_added]
                else:
                    # fallback: 直接取x之前的部分
                    header_tokens = content_tokens[:x_position]
            else:
                # 直接从开始到x位置
                header_tokens = content_tokens[:x_position]
            
            # 额外验证：如果header太长（包含system message），尝试提取纯role部分
            header_decoded = tokenizer.decode(header_tokens)
            
            # 如果包含system message，尝试只取最后的role部分
            if f"<|im_start|>{role}" in header_decoded:
                # 找到最后一个role标记的位置
                role_marker = f"<|im_start|>{role}\n"
                role_tokens = tokenizer.encode(role_marker, add_special_tokens=False)
                
                # 在header_tokens中找到role_tokens的位置
                role_pos = _find_first_subseq(header_tokens, role_tokens)
                if role_pos is not None:
                    # 只取role标记部分
                    header_tokens = role_tokens
            return header_tokens
            
    except Exception as e:
        # 不要fallback，直接报错
        raise RuntimeError(f"Failed to extract header tokens for role '{role}': {e}") from e

def parse_response_ids_to_steps(
    response_ids: List[int],
    tokenizer,
    assistant_tpl: List[int] = None,
    user_tpl: List[int] = None,
    mark_observation: bool = False,
) -> StepParseResult:
    # 1) 自动提取模板
    if assistant_tpl is None:
        assistant_tpl = _extract_role_header_tokens(tokenizer, "assistant")
    if user_tpl is None:
        user_tpl = _extract_role_header_tokens(tokenizer, "user")

    # 2) 定位 header 与 body
    a_hdr = _locate_template_positions(response_ids, assistant_tpl) if assistant_tpl else []
    u_hdr = _locate_template_positions(response_ids, user_tpl) if user_tpl else []

    a_body = [p + len(assistant_tpl) for p in a_hdr] if assistant_tpl else []
    u_body = [p + len(user_tpl) for p in u_hdr] if user_tpl else []

    # 若序列开头没有任何 header，则视为从 assistant 内容开始
    if response_ids:
        first_hdr = min(a_hdr[0] if a_hdr else len(response_ids),
                        u_hdr[0] if u_hdr else len(response_ids))
        if first_hdr > 0:
            a_hdr = [0] + a_hdr         # 伪 header：用于结束边界
            a_body = [0] + a_body       # 伪 body：用于开始边界

    # 以“header 起点”作为切分的结束边界
    cut_bounds = sorted(a_hdr + u_hdr + [len(response_ids)])

    def next_cut(pos: int) -> int:
        for b in cut_bounds:
            if b > pos:
                return b
        return len(response_ids)

    # 3) 按 body→(下一 header 起点) 构造 segments（不会吞到下个 header 的 "user"/"assistant"）
    segs = []
    for s in a_body:
        e = next_cut(s)
        if s < e:
            segs.append({"role": "assistant", "start": s, "end": e, "tokens": response_ids[s:e]})
    for s in u_body:
        e = next_cut(s)
        if s < e:
            segs.append({"role": "user", "start": s, "end": e, "tokens": response_ids[s:e]})
    segs.sort(key=lambda x: x["start"])

    if not segs:
        return StepParseResult([], [], [-1] * len(response_ids))

    # 4) 合并相邻同 role 段
    merged = []
    for seg in segs:
        if merged and merged[-1]["role"] == seg["role"] and merged[-1]["end"] == seg["start"]:
            merged[-1]["end"] = seg["end"]
            merged[-1]["tokens"].extend(seg["tokens"])
        else:
            merged.append({
                "role": seg["role"], "start": seg["start"], "end": seg["end"],
                "tokens": seg["tokens"].copy()
            })

    # 丢弃开头非 assistant 的段
    while merged and merged[0]["role"] != "assistant":
        merged.pop(0)
    if not merged:
        return StepParseResult([], [], [-1] * len(response_ids))

    # 5) 组 step（assistant 段 + 中间若干 user 段汇成 observation）
    steps = []
    i = 0
    while i < len(merged):
        a = merged[i]
        if a["role"] != "assistant":
            i += 1
            continue
        action_start, action_end = a["start"], a["end"]
        action_tokens = a["tokens"]
        action_text = tokenizer.decode(action_tokens, skip_special_tokens=True)

        j = i + 1
        obs_start = action_end
        obs_end = obs_start
        obs_tokens = []
        while j < len(merged) and merged[j]["role"] != "assistant":
            obs_end = merged[j]["end"]
            obs_tokens.extend(merged[j]["tokens"])
            j += 1
        obs_text = tokenizer.decode(obs_tokens, skip_special_tokens=True) if obs_tokens else ""

        steps.append({
            "action_tokens": action_tokens,
            "observation_tokens": obs_tokens,
            "action_text": action_text,
            "observation_text": obs_text,
            "action_start": action_start, "action_end": action_end,
            "obs_start": obs_start, "obs_end": obs_end,
        })
        i = j

    # 6) 原位打 step_ids
    step_ids = [-1] * len(response_ids)
    for k, st in enumerate(steps):
        for pos in range(st["action_start"], st["action_end"]):
            step_ids[pos] = k
        if mark_observation and st["obs_start"] < st["obs_end"]:
            for pos in range(st["obs_start"], st["obs_end"]):
                step_ids[pos] = k

    return StepParseResult(merged, steps, step_ids)


# 添加验证函数
def verify_step_alignment(batch, tokenizer, global_step):
    """验证语义评估和advantage scaling的step对齐"""
    print(f"\n=== Step Alignment Check (Step {global_step}) ===")
    
    batch_size = len(batch.batch["prompts"])
    alignment_errors = 0
    
    for sample_idx in range(min(5, batch_size)):  # 检查前5个样本
        # 从语义评估获取的steps
        semantic_steps = batch.non_tensor_batch["steps"][sample_idx]
        
        # 从step_ids获取的step数量
        step_ids = batch.batch["step_ids"][sample_idx]
        max_step_id = int(step_ids.max().item()) if (step_ids >= 0).any() else -1
        advantage_steps = max_step_id + 1 if max_step_id >= 0 else 0
        
        # 检查对齐
        semantic_count = len(semantic_steps)
        if semantic_count != advantage_steps:
            print(f"❌ Sample {sample_idx}: semantic={semantic_count}, advantage={advantage_steps}")
            alignment_errors += 1
        else:
            print(f"✅ Sample {sample_idx}: {semantic_count} steps aligned")
    
    if alignment_errors == 0:
        print("✅ [Alignment Great] All checked samples have aligned step counts!")
        return True
    else:
        print(f"❌ [Alignment Error] Found {alignment_errors} alignment errors!")
        return False
    
def verify_step_content(batch, tokenizer, sample_idx=0):
    """验证step内容的一致性"""
    print(f"\n=== Step Content Check (Sample {sample_idx}) ===")
    
    # 从batch获取
    response_tokens = batch.batch["responses"][sample_idx].tolist()
    step_ids = batch.batch["step_ids"][sample_idx].tolist()
    semantic_steps = batch.non_tensor_batch["steps"][sample_idx]
    
    # 重新解析验证
    from beyondagent.utils.step_parser import parse_response_ids_to_steps
    parse_result = parse_response_ids_to_steps(response_tokens, tokenizer)
    
    print(f"Parsed {len(parse_result.steps)} steps:")
    for i, step in enumerate(parse_result.steps):
        semantic_step = semantic_steps[i] if i < len(semantic_steps) else {"action": "MISSING", "observation": "MISSING"}
        print(f"Step {i}:")
        print(f"  Parsed Action: {step['action_text'][:50]}...")
        print(f"  Semantic Action: {semantic_step.get('action', 'MISSING')[:50]}...")
        print(f"  Match: {step['action_text'].strip() == semantic_step.get('action', '').strip()}")


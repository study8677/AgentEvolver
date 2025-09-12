import torch
import verl.utils.torch_functional as verl_F
from openai import AsyncOpenAI
import os
import json
from pathlib import Path
from loguru import logger
import time
import traceback
from tqdm import tqdm
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional, Literal
import threading
from dataclasses import dataclass, asdict
import random
from beyondagent.module.credit_manager.prompt import build_batch_adv_evaluation_prompt, build_batch_reward_evaluation_prompt

__all__ = [
    "evaluate_step_flags_parallel",
    "ParallelSemanticProcessor",
]

@dataclass
class EvaluationTask:
    """评估任务的数据结构 - 一次评估整个sample的所有steps"""
    sample_idx: int
    query: str
    rollout: str
    steps: List[Dict[str, str]]  # ← 原来是 List[str]，统一为 parse_rollout_to_steps 的结构
    overall_score: float

@dataclass
class EvaluationResult:
    """评估结果的数据结构 - 包含整个sample的所有step结果"""
    sample_idx: int
    step_results: List[bool]  # 所有steps的评估结果
    response_time: float

@dataclass
class EvaluationRecord:
    """评估记录的数据结构，用于保存到文件"""
    sample_idx: int
    query: str
    rollout: str
    steps: List[str]
    overall_score: float
    llm_input_messages: List[Dict]
    llm_raw_output: str
    llm_parsed_results: List[bool]  # 所有steps的解析结果
    response_time: float
    timestamp: float
    model_name: str
    evaluation_type: str
    global_step: Optional[int] = None
    epoch: Optional[str] = None

# =========================================================
# Added: rollout parsing & batch-eval prompt utilities
# =========================================================
import re 

def _steps_struct_to_text_list(steps: List[Dict[str, str]]) -> List[str]:
    out = []
    for st in steps:
        act = (st.get("action") or "").strip()
        obs = (st.get("observation") or "").strip()
        if obs:
            out.append(f"{act}\n\n[OBSERVATION]\n{obs}")
        else:
            out.append(act)
    return out




def parse_batch_evaluation_result(response: str, num_steps: int):
    numbered = {}
    for m in re.finditer(r"Step\s+(\d+)\s+Judgment:\s*(GOOD|BAD)", response, flags=re.I):
        numbered[int(m.group(1))] = m.group(2).upper() == "GOOD"
    if len(numbered) == num_steps:
        return [numbered[i] for i in range(num_steps)]
    flags = re.findall(r"\b(GOOD|BAD)\b", response.upper())
    if len(flags) >= num_steps:
        return [flag == "GOOD" for flag in flags[:num_steps]]
    raise ValueError("Could not parse evaluation result")

def _get_overall_advantage(advantages_tensor, mask=None):
    """从advantages tensor中获取overall advantage值"""
    if advantages_tensor.dim() == 0:
        return advantages_tensor.item()
    
    if advantages_tensor.dim() == 1:
        if mask is not None:
            valid_advantages = advantages_tensor[mask.bool()]
            if len(valid_advantages) > 0:
                return valid_advantages[0].item()
            else:
                return 0.0
        else:
            non_zero_mask = torch.abs(advantages_tensor) > 1e-8
            if non_zero_mask.any():
                return advantages_tensor[non_zero_mask][0].item()
            else:
                return 0.0
    
    raise ValueError(f"Unsupported advantages_tensor shape: {advantages_tensor.shape}")

def _save_evaluation_record(record: EvaluationRecord, save_dir: Optional[str] = None):
    """保存评估记录到文件"""
    if save_dir is None:
        return
    
    try:
        base_save_path = Path(save_dir)
        base_save_path.mkdir(parents=True, exist_ok=True)
        
        if record.global_step is not None:
            step_subdir = f"step_{record.global_step:06d}"
        else:
            step_subdir = "step_unknown"
        
        step_save_path = base_save_path / step_subdir
        step_save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp_str = f"{record.timestamp:.3f}".replace('.', '_')
        global_step_str = f"step{record.global_step:06d}" if record.global_step is not None else "nostep"
        filename = f"{global_step_str}_sample{record.sample_idx:03d}_{timestamp_str}.json"
        
        file_path = step_save_path / filename
        record_dict = asdict(record)
        record_dict["_metadata"] = {
            "save_time": time.time(),
            "step_directory": step_subdir,
            "file_name": filename,
            "full_path": str(file_path),
            "num_steps": len(record.steps)
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(record_dict, f, ensure_ascii=False, indent=2)
        
        print(f"[record_save] ✅ Saved sample {record.sample_idx} with {len(record.steps)} steps: {step_subdir}/{filename}")
            
    except Exception as e:
        print(f"[record_save] ❌ FAILED to save evaluation record for sample {record.sample_idx}: {e}")
        print(f"[record_save] 📁 Path: {save_dir}")

# ————————————————————————————————————————————————————————————————
# API评估 - 增强的重试机制
# ————————————————————————————————————————————————————————————————

async def _async_safe_query(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    semaphore: asyncio.Semaphore,
    max_retries: int = 200,
    timeout_s: int = 120,         # 可调：单次请求超时阈值
) -> str:
    """异步 API 调用：对 **所有** 异常重试，429 用指数退避"""
    async with semaphore:
        last_exception = None

        for attempt in range(max_retries):
            try:
                # ---------- 普通 / thinking 模型分支 ----------
                is_thinking_model = model.lower() in {
                    "qwq-plus",
                    "qwen3-30b-a3b-thinking-2507",
                    "qwen3-235b-a22b-thinking-2507",
                }

                if is_thinking_model:
                    print(f"[API] Using streaming mode for thinking model: {model}")
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.0,
                        extra_body={"enable_thinking": True},
                        stream=True,
                        max_tokens=8192,
                        timeout=timeout_s,
                    )

                    answer_content, reasoning_content = "", ""
                    async for chunk in response:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        if getattr(delta, "reasoning_content", None):
                            reasoning_content += delta.reasoning_content
                        if getattr(delta, "content", ""):
                            answer_content += delta.content

                    final_answer = answer_content.strip()
                    return final_answer

                else:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.0,
                        timeout=timeout_s,
                        max_tokens=8192,
                    )
                    return response.choices[0].message.content.strip()

            # ---------- 统一异常处理 ----------
            except Exception as e:                # ❶ 捕获所有异常
                last_exception = e
                err = str(e).lower()

                is_rate_limit = any(
                    key in err
                    for key in [
                        "429",
                        "rate limit",
                        "exceeded your current requests",
                        "limit_requests",
                    ]
                )

                # ----------- 若未到最大重试次数 -----------
                if attempt < max_retries - 1:
                    # 429 → 指数退避 + 抖动
                    if is_rate_limit:
                        backoff = min(1.5 ** attempt, 60)       # 上限 60 s
                        jitter  = backoff * 0.25 * random.random()
                        wait    = backoff + jitter
                        print(f"[API Retry] 429 (attempt {attempt+1}/{max_retries}) "
                              f"sleep {wait:.1f}s")
                        await asyncio.sleep(wait)
                    else:
                        # 其它异常 → 线性退避
                        wait = min(2.0 * (attempt + 1), 15)
                        print(f"[API Retry] {type(e).__name__}: {e} "
                              f"(attempt {attempt+1}/{max_retries}) sleep {wait:.1f}s")
                        await asyncio.sleep(wait)
                    # 继续 for-loop
                else:
                    # ----------- 已达到最大重试次数 -----------
                    print(f"[API Error] ❌ Max retries ({max_retries}) exceeded: {e}")
                    break

        # loop 结束仍失败 → 抛出最后一次异常
        raise last_exception


async def _evaluate_single_sample_api(
    client: AsyncOpenAI,
    model_name: str,
    task: EvaluationTask,
    semaphore: asyncio.Semaphore,
    overall_score_source: str = "advantages", 
    max_retries: int = 200,
    save_dir: Optional[str] = None,
    global_step: Optional[int] = None,
    epoch: Optional[str] = None
) -> EvaluationResult:
    """使用 API 一次性评估单个 sample 的所有 steps"""
    start_time = time.time()

    try:
        # 1) 构造批量评估 prompt
        # shuchang: 0809
        # FIXME: 这里组织prompt改为直接用 steps 结构
        if overall_score_source == "token_level_rewards":
            messages = build_batch_reward_evaluation_prompt(
                task.query, task.steps, task.overall_score
            )
        elif overall_score_source == "advantages":
            messages = build_batch_adv_evaluation_prompt(
                task.query, task.steps, task.overall_score
            )
        # 2) 调用 LLM
        llm_raw_output = await _async_safe_query(
            client, model_name, messages, semaphore, max_retries
        )

        # 3) 解析结果
        try:
            step_results = parse_batch_evaluation_result(
                llm_raw_output, len(task.steps)
            )
            print(
                f"[API] ✅ Sample {task.sample_idx}: Successfully parsed "
                f"{len(step_results)} step results"
            )
        except Exception as parse_error:
            # ——> 解析失败：不做任何缩放（全部使用 “无缩放” 标记）
            print(
                f"[API] ❌ Sample {task.sample_idx}: Parse error, "
                f"disable rescale: {parse_error}"
            )
            uniform_flag = task.overall_score > 0  # True=GOOD, False=BAD
            step_results = [uniform_flag for _ in task.steps]

        response_time = time.time() - start_time

        # 4) 保存评估记录（保持不变）
        if save_dir:
            is_thinking_model = model_name.lower() in {
                "qwq-plus",
                "qwen3-30b-a3b-thinking-2507",
                "qwen3-235b-a22b-thinking-2507",
            }
            record = EvaluationRecord(
                sample_idx=task.sample_idx,
                query=task.query,
                rollout=task.rollout,
                steps=_steps_struct_to_text_list(task.steps),  # ← 关键
                overall_score=task.overall_score,
                llm_input_messages=messages,
                llm_raw_output=llm_raw_output,
                llm_parsed_results=step_results,
                response_time=response_time,
                timestamp=time.time(),
                model_name=f"{model_name}{'_thinking' if is_thinking_model else ''}",
                evaluation_type="api",
                global_step=global_step,
                epoch=epoch,
            )
            _save_evaluation_record(record, save_dir)

        return EvaluationResult(
            sample_idx=task.sample_idx,
            step_results=step_results,
            response_time=response_time,
        )

    except Exception as e:
        # ——> API 整体失败：同样不做任何缩放
        response_time = time.time() - start_time
        print(f"[parallel_eval] ❌ FAILED to evaluate sample {task.sample_idx}: {e}")

        uniform_flag = task.overall_score > 0
        step_results = [uniform_flag for _ in task.steps]

        if save_dir:
            record = EvaluationRecord(
                sample_idx=task.sample_idx,
                query=task.query,
                rollout=task.rollout,
                steps=task.steps,
                overall_score=task.overall_score,
                llm_input_messages=[],
                llm_raw_output=f"ERROR: {str(e)}",
                llm_parsed_results=step_results,
                response_time=response_time,
                timestamp=time.time(),
                model_name=model_name,
                evaluation_type="api",
                global_step=global_step,
                epoch=epoch,
            )
            _save_evaluation_record(record, save_dir)

        return EvaluationResult(
            sample_idx=task.sample_idx,
            step_results=step_results,
            response_time=response_time,
        )

# ————————————————————————————————————————————————————————————————
# 统一的并行评估接口
# ————————————————————————————————————————————————————————————————

async def evaluate_step_flags_parallel(tokenizer, batch, overall_score_source: str = "advantages",  model_name: str = "qwen-max", evaluation_type: Literal["api"] = "api", max_concurrent: int = 20, batch_size_limit: int = 100, mask_tensor: torch.Tensor = None, api_max_retries: int = 200, save_dir: Optional[str] = None, global_step: Optional[int] = None, epoch: Optional[str] = None, skip_type: str='skip_small_adv') -> Tuple[List[List[bool]], Dict]:
    """并行评估step flags，每个sample一次API调用评估所有steps
    NOTE: SSA中根据advantage评估 和 PRM-GRPO中根据ORM评估均可使用本函数
    """
    batch_size = len(batch.batch['prompts'])
    print(f"[parallel_eval] Starting parallel evaluation for {batch_size} samples using API mode")
    print(f"[parallel_eval] 🚀 OPTIMIZED: One API call per sample (not per step)")
    print(f"[parallel_eval] Model: {model_name}, API max retries: {api_max_retries}")
    if save_dir:
        print(f"[parallel_eval] Saving evaluation records to: {save_dir}")
    
    if 'steps' not in batch.non_tensor_batch:
        raise ValueError("❌ batch.non_tensor_batch['steps'] is required but not found")
    
    if evaluation_type != "api":
        raise ValueError(f"❌ Only 'api' evaluation_type is supported, got: {evaluation_type}")
    
    # 初始化API客户端
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ [parallel_eval] No API key found in DASHSCOPE_API_KEY environment variable")
        print("❌ [parallel_eval] Please set: export DASHSCOPE_API_KEY='your-api-key'")
        print("❌ [parallel_eval] Using random fallback for evaluation")
        # shuchang: 0809
        # FIXME: 注释掉fallback，强制要求必须有API KEY
        # return _apply_fallback_strategy_parallel(batch, tokenizer), {"fallback_used": True, "evaluation_type": evaluation_type}
        raise  RuntimeError("No API key found in DASHSCOPE_API_KEY environment variable")
    
    api_client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    # 🚀 关键优化：按sample创建任务，而不是按step
    all_tasks = []
    flags_per_sample = [[] for _ in range(batch_size)]
    skipped_samples = 0
    
    if mask_tensor is not None:
        response_mask = mask_tensor
        print(f"[parallel_eval] Using external mask tensor with shape {mask_tensor.shape}")
        
        response_length = batch.batch["responses"].size(1)
        if response_mask.shape != (batch_size, response_length):
            raise ValueError(f"❌ mask_tensor shape {response_mask.shape} doesn't match expected shape ({batch_size}, {response_length})")
    else:
        response_length = batch.batch["responses"].size(1)
        response_mask = batch.batch["loss_mask"][:, -response_length:]
        print(f"[parallel_eval] Using default loss_mask")

    for sample_idx in range(batch_size):
        query = tokenizer.decode(batch.batch["prompts"][sample_idx], skip_special_tokens=True)
        rollout = tokenizer.decode(batch.batch["responses"][sample_idx], skip_special_tokens=True)
        # shuchang: 0809
        # FIXME: 这里改为直接用 batch.non_tensor_batch["steps"]，不需要再额外解析
        # steps_struct = parse_rollout_to_steps(rollout)
        steps_struct = batch.non_tensor_batch["steps"][sample_idx]

        # mask 与 overall_score 维持原逻辑
        sample_mask = response_mask[sample_idx]
        advantage = _get_overall_advantage(batch.batch["advantages"][sample_idx], sample_mask)
        orm_reward = batch.batch["token_level_rewards"][sample_idx].sum().item()
        if overall_score_source == "token_level_rewards":
            # PRM-GRPO 模式：使用原始 ORM 奖励
            # shuchang 0904：reward 0,1 映射为-1,1
            orm_scores = 1.0 if orm_reward > 0 else -1.0
            overall_score = orm_scores
        elif overall_score_source == "advantages":
            # SSA 模式：使用计算后的 advantage
            overall_score = advantage
        else:
            overall_score = orm_scores
        # shuchang: 0906
        # 只跳过 advantage 非常小的样本 或 全部为负的样本
        # 决定是否应该跳过当前样本
        should_skip = False
        skip_reason = ""
        
        if skip_type == "skip_small_adv":
            # 1. 只跳过 advantage 非常小的样本
            if abs(advantage) < 1e-8:
                should_skip = True
                skip_reason = f"advantage≈0 ({advantage:.6f})"
        
        elif skip_type == "skip_all_neg":
            # 2. 跳过 orm_reward 为负或零的样本
            # 注意：orm_reward > 0 才是正样本，所以 <= 0 都属于“负”的范畴
            if orm_reward <= 0:
                should_skip = True
                skip_reason = f"orm_reward is not positive ({orm_reward:.6f})"

        # 如果满足任一跳过条件，则执行跳过逻辑
        if should_skip:
            print(f"[parallel_eval] Sample {sample_idx}: Skipping evaluation due to {skip_reason}. Assigning flags based on overall_score.")
            # 根据 overall_score 的正负来决定 flag 的值
            flag_value = overall_score > 0
            flags_per_sample[sample_idx] = [flag_value] * len(steps_struct)

            if save_dir:
                record = EvaluationRecord(
                    sample_idx=sample_idx,
                    query=query,
                    rollout=rollout,
                    # ✅ 日志里仍按原来的 List[str] 存
                    steps=_steps_struct_to_text_list(steps_struct),
                    overall_score=overall_score,
                    llm_input_messages=[],
                    llm_raw_output="SKIPPED_ZERO_ADVANTAGE",
                    llm_parsed_results=[True] * len(steps_struct),
                    response_time=0.0,
                    timestamp=time.time(),
                    model_name=model_name,
                    evaluation_type=evaluation_type,
                    global_step=global_step,
                    epoch=epoch
                )
                _save_evaluation_record(record, save_dir)
            skipped_samples += 1
            continue

        
       # ✅ EvaluationTask 使用结构化 steps
        task = EvaluationTask(
            sample_idx=sample_idx,
            query=query,
            rollout=rollout,
            steps=steps_struct,
            overall_score=overall_score
        )
        all_tasks.append(task)
    
    total_tasks = len(all_tasks)
    total_api_calls = total_tasks  # 现在每个sample只需要一次API调用
    total_steps = sum(len(t.steps) for t in all_tasks)
    # --- 指标准备：每个样本的step长度 ---
    step_len_map = {t.sample_idx: len(t.steps) for t in all_tasks}
    step_len_list = list(step_len_map.values())
    
    print(f"[parallel_eval] 🚀 EFFICIENCY GAIN:")
    print(f"[parallel_eval]   - Total samples: {batch_size}")
    print(f"[parallel_eval]   - Total steps: {total_steps}")
    print(f"[parallel_eval]   - API calls needed: {total_api_calls} (instead of {total_steps})")
    print(f"[parallel_eval]   - Efficiency gain: {total_steps/max(1,total_api_calls):.1f}x fewer API calls")
    print(f"[parallel_eval]   - Skipped {skipped_samples} samples with advantage=0")
    
    if total_tasks == 0:
        print("[parallel_eval] No tasks to process, all samples had advantage=0")
        await api_client.close()
        return flags_per_sample, {
            "total_tasks": 0,
            "total_api_calls": 0,
            "total_steps": total_steps,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "total_api_time": 0,
            "avg_api_time": 0,
            "max_concurrent": max_concurrent,
            "fallback_used": False,
            "skipped_samples": skipped_samples,
            "evaluation_type": evaluation_type,
            "api_max_retries": api_max_retries,
            "efficiency_gain": 0
        }
    
    all_results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    with tqdm(total=total_tasks, desc=f"[parallel_eval] Processing samples (API)") as pbar:
        for i in range(0, total_tasks, batch_size_limit):
            batch_tasks = all_tasks[i:i + batch_size_limit]
            
            # 每个task调用_evaluate_single_sample_api，一次性评估整个sample的所有steps
            coroutines = [
                _evaluate_single_sample_api(api_client, model_name, task, semaphore, overall_score_source, api_max_retries, save_dir, global_step, epoch)
                for task in batch_tasks
            ]
            batch_results = await asyncio.gather(*coroutines, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"[parallel_eval] ❌ Task failed with exception: {result}")
                    continue
                all_results.append(result)
            
            pbar.update(len(batch_tasks))
    
    # 整理结果到flags_per_sample
    for result in all_results:
        flags_per_sample[result.sample_idx] = result.step_results
    
    total_time = sum(r.response_time for r in all_results)
    avg_time = total_time / len(all_results) if all_results else 0
    
    stats = {
        "total_tasks": total_tasks,
        "total_api_calls": len(all_results),
        "total_steps": total_steps,
        "successful_tasks": len(all_results),
        "failed_tasks": total_tasks - len(all_results),
        "total_api_time": total_time,
        "avg_api_time": avg_time,
        "max_concurrent": max_concurrent,
        "fallback_used": False,
        "skipped_samples": skipped_samples,
        "evaluation_type": evaluation_type,
        "model_name": model_name,
        "api_max_retries": api_max_retries,
        "save_dir": save_dir,
        "efficiency_gain": total_steps / max(1, len(all_results))  # 效率提升倍数
    }
    def _p95(vals):
        if not vals:
            return 0.0
        s = sorted(vals)
        k = int(round(0.95 * (len(s) - 1)))
        return float(s[k])

    parsed_ok = sum(
        1 for r in all_results
        if len(r.step_results) == step_len_map.get(r.sample_idx, 0)
    )
    length_mismatch = sum(
        1 for r in all_results
        if len(r.step_results) != step_len_map.get(r.sample_idx, 0)
    )

    stats.update({
        "prm/parse_success_rate": parsed_ok / max(1, total_tasks),
        "prm/avg_steps_per_sample": (sum(step_len_list) / max(1, len(step_len_list))) if step_len_list else 0.0,
        "prm/p95_steps_per_sample": _p95(step_len_list),
        "prm/flags_len_mismatch_rate": length_mismatch / max(1, total_tasks),
        # 可选：原始计数，便于排错
        "prm/_parse_success_count": parsed_ok,
        "prm/_flags_len_mismatch_count": length_mismatch,
    })
    
    print(f"[parallel_eval] ✅ Completed with {stats['efficiency_gain']:.1f}x efficiency gain!")
    print(f"[parallel_eval] Stats: {stats}")
    
    await api_client.close()
    return flags_per_sample, stats


def evaluate_step_flags_parallel_sync(tokenizer, batch, **kwargs):
    """evaluate_step_flags_parallel的同步包装器"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        evaluate_step_flags_parallel(tokenizer, batch, **kwargs)
    )
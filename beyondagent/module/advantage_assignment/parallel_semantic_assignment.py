import torch
import verl.utils.torch_functional as verl_F
from openai import AsyncOpenAI
import os
from loguru import logger
import time
import traceback
from tqdm import tqdm
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional, Literal
import threading
from dataclasses import dataclass

__all__ = [
    "evaluate_step_flags_parallel",    # 并行版本的step评估
    "apply_step_mask_vectorized",      # 向量化的mask应用
    "ParallelSemanticProcessor",       # 统一的处理器类
]

@dataclass
class EvaluationTask:
    """评估任务的数据结构"""
    sample_idx: int
    step_idx: int
    query: str
    rollout: str
    step_text: str
    overall_adv: float

@dataclass
class EvaluationResult:
    """评估结果的数据结构"""
    sample_idx: int
    step_idx: int
    is_good: bool
    response_time: float

# 全局变量存储vLLM模型和tokenizer（用于本地评估）
_vllm_model = None
_vllm_tokenizer = None
_model_lock = threading.Lock()

def _get_overall_advantage(advantages_tensor, mask=None):
    """
    从advantages tensor中获取overall advantage值
    在GRPO中，所有有效token共享一个advantage，我们需要正确提取这个值
    
    Args:
        advantages_tensor: advantage tensor, shape (resp_len,) 
        mask: 标识需要训练的token位置的mask，shape (resp_len,)
              可以是loss_mask或response_mask，取决于外部传入
    
    Returns:
        float: 提取到的overall advantage值
    """
    if advantages_tensor.dim() == 0:  # scalar
        return advantages_tensor.item()
    
    if advantages_tensor.dim() == 1:  # shape: (resp_len,)
        # 优先使用mask来提取有效advantage
        if mask is not None:
            valid_advantages = advantages_tensor[mask.bool()]
            if len(valid_advantages) > 0:
                # 在GRPO中，所有有效token的advantage应该相同，取第一个即可
                return valid_advantages[0].item()
            else:
                # mask中没有有效token，返回0
                return 0.0
        else:
            # fallback: 没有mask时，寻找第一个非零值
            non_zero_mask = torch.abs(advantages_tensor) > 1e-8
            if non_zero_mask.any():
                return advantages_tensor[non_zero_mask][0].item()
            else:
                return 0.0
    
    # 其他维度不支持
    raise ValueError(f"Unsupported advantages_tensor shape: {advantages_tensor.shape}")

def _build_prompt(query: str, rollout: str, step: str, overall_adv: float) -> list[dict]:
    """
    构造对话消息（与原版相同）
    
    Args:
        overall_adv: 真正的共享advantage值（GRPO中所有token共享），
                    不是sum()后被序列长度放大的错误值
    """
    polarity = "positive" if overall_adv > 0 else "negative"
    sys = "You are an expert reward-model evaluator. Reply with **exactly one word**, either **GOOD** or **BAD** – no explanations."
    user = (
        f"────────────────────────────────\n"
        f"USER QUERY\n{query}\n\n"
        f"ASSISTANT FULL ANSWER\n{rollout}\n\n"
        f"CURRENT ASSISTANT STEP\n{step}\n"
        f"────────────────────────────────\n\n"
        f"The total advantage (quality score) of the full answer is "
        f"**{overall_adv:+.4f}** → this is {polarity} "
        f"(positive if > 0, negative if < 0).\n\n"
        f"**Task**\n"
        f"Does the *current assistant step* improve (GOOD) or harm (BAD) "
        f"the final answer given the user query and the overall advantage?"
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

# ————————————————————————————————————————————————————————————————
# 本地模型评估（vLLM）
# ————————————————————————————————————————————————————————————————

def _initialize_vllm_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """初始化vLLM模型和tokenizer（线程安全）"""
    global _vllm_model, _vllm_tokenizer
    
    with _model_lock:
        if _vllm_model is None:
            print(f"[vLLM] Initializing model: {model_name}")
            try:
                # 新增：vLLM相关导入
                from transformers import AutoTokenizer
                from vllm import LLM, SamplingParams
                
                # 初始化tokenizer
                _vllm_tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # 初始化vLLM引擎
                _vllm_model = LLM(
                    model=model_name,
                    gpu_memory_utilization=0.8,  # 调整GPU内存使用率
                    tensor_parallel_size=1,      # 单GPU
                    dtype="auto",                # 自动选择数据类型
                    trust_remote_code=True
                )
                print(f"[vLLM] Model initialized successfully")
            except Exception as e:
                print(f"[vLLM] Failed to initialize model: {e}")
                raise e
    
    return _vllm_model, _vllm_tokenizer

async def _vllm_safe_query(model, 
                          tokenizer,
                          messages: list[dict], 
                          semaphore: asyncio.Semaphore,
                          max_retries: int = 3) -> str:
    """使用vLLM进行安全的本地推理"""
    async with semaphore:  # 控制并发数
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # 转换消息格式为文本
                text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True,
                    enable_thinking=False  # 对于简单的GOOD/BAD判断，不需要thinking模式
                )
                
                # 配置采样参数（为了获得确定性结果）
                from vllm import SamplingParams
                sampling_params = SamplingParams(
                    temperature=0.0,     # 确定性输出
                    top_p=1.0,
                    top_k=-1,
                    max_tokens=10,       # 只需要一个词
                    stop=None
                )
                
                # 在线程池中运行同步的vLLM推理
                def run_vllm():
                    outputs = model.generate([text], sampling_params)
                    return outputs[0].outputs[0].text.strip()
                
                # 使用asyncio.to_thread将同步调用转为异步
                result = await asyncio.to_thread(run_vllm)
                
                return result
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
        
        raise last_exception

async def _evaluate_single_task_vllm(model,
                                    tokenizer,
                                    task: EvaluationTask,
                                    semaphore: asyncio.Semaphore) -> EvaluationResult:
    """使用vLLM评估单个任务"""
    start_time = time.time()
    
    try:
        messages = _build_prompt(task.query, task.rollout, task.step_text, task.overall_adv)
        answer = await _vllm_safe_query(model, tokenizer, messages, semaphore)
        
        answer_upper = answer.upper()
        is_good = answer_upper.startswith("G") or "GOOD" in answer_upper
        
        response_time = time.time() - start_time
        
        return EvaluationResult(
            sample_idx=task.sample_idx,
            step_idx=task.step_idx,
            is_good=is_good,
            response_time=response_time
        )
        
    except Exception as e:
        response_time = time.time() - start_time
        print(f"[parallel_eval] Failed to evaluate sample {task.sample_idx}, step {task.step_idx}: {e}")
        
        # 失败时使用随机fallback
        import random
        is_good = random.choice([True, False])
        
        return EvaluationResult(
            sample_idx=task.sample_idx,
            step_idx=task.step_idx,
            is_good=is_good,
            response_time=response_time
        )

# ————————————————————————————————————————————————————————————————
# API评估（OpenAI兼容）- 增强的重试机制
# ————————————————————————————————————————————————————————————————

async def _async_safe_query(client: AsyncOpenAI, 
                           model: str, 
                           messages: list[dict], 
                           semaphore: asyncio.Semaphore,
                           max_retries: int = 200) -> str:
    """
    异步安全的API调用，增强的重试机制，专门处理429错误
    
    Args:
        client: OpenAI客户端
        model: 模型名称
        messages: 消息列表
        semaphore: 并发控制信号量
        max_retries: 最大重试次数，默认200次
    
    Returns:
        API响应内容
    """
    async with semaphore:  # 控制并发数
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    timeout=30,
                    max_tokens=10,
                )
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # 检查是否是429错误
                is_rate_limit_error = (
                    "429" in error_str or 
                    "rate limit" in error_str or
                    "limit_requests" in error_str or
                    "exceeded your current requests" in error_str
                )
                
                # 检查是否是其他可重试的错误
                is_retryable_error = (
                    "timeout" in error_str or
                    "connection" in error_str or
                    "500" in error_str or
                    "502" in error_str or
                    "503" in error_str or
                    "504" in error_str
                )
                
                if attempt < max_retries - 1:  # 不是最后一次尝试
                    if is_rate_limit_error:
                        # 429错误：使用指数退避，但有上限
                        # 基础等待时间：1秒，每次翻倍，最大60秒
                        base_wait = min(1.0 * (2 ** min(attempt, 6)), 60.0)
                        # 添加随机抖动，避免所有请求同时重试
                        import random
                        jitter = random.uniform(0.1, 0.3) * base_wait
                        wait_time = base_wait + jitter
                        
                        print(f"[API Retry] 429 Rate limit hit, attempt {attempt + 1}/{max_retries}, waiting {wait_time:.2f}s")
                        await asyncio.sleep(wait_time)
                        
                    elif is_retryable_error:
                        # 其他可重试错误：较短的等待时间
                        wait_time = min(2.0 * (attempt + 1), 10.0)
                        print(f"[API Retry] Retryable error, attempt {attempt + 1}/{max_retries}, waiting {wait_time:.2f}s: {e}")
                        await asyncio.sleep(wait_time)
                        
                    else:
                        # 不可重试的错误，立即失败
                        print(f"[API Error] Non-retryable error, failing immediately: {e}")
                        break
                else:
                    # 最后一次尝试失败
                    if is_rate_limit_error:
                        print(f"[API Error] Rate limit exceeded after {max_retries} attempts")
                    else:
                        print(f"[API Error] Max retries ({max_retries}) exceeded: {e}")
        
        raise last_exception

async def _evaluate_single_task_api(client: AsyncOpenAI,
                                  model_name: str,
                                  task: EvaluationTask,
                                  semaphore: asyncio.Semaphore,
                                  max_retries: int = 200) -> EvaluationResult:
    """
    使用API评估单个任务，增强重试机制
    
    Args:
        client: OpenAI客户端
        model_name: 模型名称
        task: 评估任务
        semaphore: 并发控制信号量
        max_retries: 最大重试次数
    """
    start_time = time.time()
    
    try:
        messages = _build_prompt(task.query, task.rollout, task.step_text, task.overall_adv)
        answer = await _async_safe_query(client, model_name, messages, semaphore, max_retries)
        
        answer_upper = answer.upper()
        is_good = answer_upper.startswith("G") or "GOOD" in answer_upper
        
        response_time = time.time() - start_time
        
        return EvaluationResult(
            sample_idx=task.sample_idx,
            step_idx=task.step_idx,
            is_good=is_good,
            response_time=response_time
        )
        
    except Exception as e:
        response_time = time.time() - start_time
        print(f"[parallel_eval] Failed to evaluate sample {task.sample_idx}, step {task.step_idx} after all retries: {e}")
        
        # 失败时使用随机fallback
        import random
        is_good = random.choice([True, False])
        
        return EvaluationResult(
            sample_idx=task.sample_idx,
            step_idx=task.step_idx,
            is_good=is_good,
            response_time=response_time
        )

# ————————————————————————————————————————————————————————————————
# 统一的并行评估接口
# ————————————————————————————————————————————————————————————————

async def evaluate_step_flags_parallel(tokenizer,
                                     batch,
                                     model_name: str = "qwen-max",
                                     evaluation_type: Literal["local", "api"] = "api",
                                     max_concurrent: int = 20,
                                     batch_size_limit: int = 100,
                                     mask_tensor: torch.Tensor = None,
                                     api_max_retries: int = 200) -> Tuple[List[List[bool]], Dict]:
    """
    并行评估step flags，支持本地模型和API两种方式
    对于advantage=0的样本跳过评估，直接返回GOOD
    
    Args:
        tokenizer: 分词器
        batch: 数据批次
        model_name: 模型名称
        evaluation_type: 评估类型，"local"使用vLLM本地模型，"api"使用API调用
        max_concurrent: 最大并发数
        batch_size_limit: 单批次处理的最大任务数
        mask_tensor: 外部传入的mask tensor，shape (bs, resp_len)
                    可以是loss_mask或response_mask，如果为None则使用默认的loss_mask
        api_max_retries: API调用的最大重试次数，特别用于处理429错误
        
    Returns:
        (flags_per_sample, stats): 评估结果和统计信息
    """
    batch_size = len(batch.batch['prompts'])
    print(f"[parallel_eval] Starting parallel evaluation for {batch_size} samples using {evaluation_type} mode")
    print(f"[parallel_eval] Model: {model_name}, API max retries: {api_max_retries}")
    
    # 检查必要的输入
    if 'steps' not in batch.non_tensor_batch:
        raise ValueError("batch.non_tensor_batch['steps'] is required but not found")
    
    # 根据评估类型初始化
    if evaluation_type == "local":
        # 初始化vLLM模型
        try:
            vllm_model, vllm_tokenizer = _initialize_vllm_model(model_name)
            api_client = None
        except Exception as e:
            print(f"[parallel_eval] Failed to initialize vLLM model, using random fallback: {e}")
            return _apply_fallback_strategy_parallel(batch), {"fallback_used": True, "error": str(e), "evaluation_type": evaluation_type}
    elif evaluation_type == "api":
        # 初始化API客户端
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            print("[parallel_eval] No API key found, using random fallback")
            return _apply_fallback_strategy_parallel(batch), {"fallback_used": True, "evaluation_type": evaluation_type}
        
        api_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        vllm_model = vllm_tokenizer = None
    else:
        raise ValueError(f"Unsupported evaluation_type: {evaluation_type}. Must be 'local' or 'api'")
    
    # 准备所有评估任务，跳过advantage=0的样本
    all_tasks = []
    flags_per_sample = [[] for _ in range(batch_size)]
    skipped_samples = 0
    
    # 🔧 关键修改：使用外部传入的mask_tensor，如果没有传入则使用默认的loss_mask
    if mask_tensor is not None:
        response_mask = mask_tensor
        print(f"[parallel_eval] Using external mask tensor with shape {mask_tensor.shape}")
        
        # 验证mask tensor的形状
        response_length = batch.batch["responses"].size(1)
        if response_mask.shape != (batch_size, response_length):
            raise ValueError(f"mask_tensor shape {response_mask.shape} doesn't match expected shape ({batch_size}, {response_length})")
    else:
        # 使用默认的loss_mask
        response_length = batch.batch["responses"].size(1)
        response_mask = batch.batch["loss_mask"][:, -response_length:]
        print(f"[parallel_eval] Using default loss_mask")

    for sample_idx in range(batch_size):
        query = tokenizer.decode(batch.batch["prompts"][sample_idx], skip_special_tokens=True)
        rollout = tokenizer.decode(batch.batch["responses"][sample_idx], skip_special_tokens=True)
        steps = batch.non_tensor_batch["steps"][sample_idx]
        
        # 使用传入的mask提取正确的overall advantage
        sample_mask = response_mask[sample_idx]
        
        overall_adv = _get_overall_advantage(
            batch.batch["advantages"][sample_idx], 
            sample_mask
        )
        
        # 新增：如果advantage为0，直接设置所有step为GOOD，跳过API调用
        if abs(overall_adv) < 1e-8:  # 使用小的阈值处理浮点精度问题
            print(f"[parallel_eval] Sample {sample_idx}: advantage≈0 ({overall_adv:.6f}), skipping evaluation, returning all GOOD")
            flags_per_sample[sample_idx] = [True] * len(steps)  # 所有step都标记为GOOD
            skipped_samples += 1
            continue
        
        # 为非零advantage的样本创建评估任务
        for step_idx, step_text in enumerate(steps):
            task = EvaluationTask(
                sample_idx=sample_idx,
                step_idx=step_idx,
                query=query,
                rollout=rollout,
                step_text=step_text,
                overall_adv=overall_adv
            )
            all_tasks.append(task)
    
    total_tasks = len(all_tasks)
    print(f"[parallel_eval] Total tasks to process: {total_tasks}")
    print(f"[parallel_eval] Skipped {skipped_samples} samples with advantage=0")
    
    if total_tasks == 0:
        # 所有样本都被跳过了
        print("[parallel_eval] No tasks to process, all samples had advantage=0")
        if api_client:
            await api_client.close()
        return flags_per_sample, {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "total_api_time": 0,
            "avg_api_time": 0,
            "max_concurrent": max_concurrent,
            "fallback_used": False,
            "skipped_samples": skipped_samples,
            "evaluation_type": evaluation_type,
            "api_max_retries": api_max_retries
        }
    
    # 分批处理任务（避免内存过大）
    all_results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # 使用进度条
    with tqdm(total=total_tasks, desc=f"[parallel_eval] Processing tasks ({evaluation_type})") as pbar:
        for i in range(0, total_tasks, batch_size_limit):
            batch_tasks = all_tasks[i:i + batch_size_limit]
            
            # 根据评估类型创建协程任务
            if evaluation_type == "local":
                coroutines = [
                    _evaluate_single_task_vllm(vllm_model, vllm_tokenizer, task, semaphore)
                    for task in batch_tasks
                ]
            else:  # api
                coroutines = [
                    _evaluate_single_task_api(api_client, model_name, task, semaphore, api_max_retries)
                    for task in batch_tasks
                ]
            
            # 等待当前批次完成
            batch_results = await asyncio.gather(*coroutines, return_exceptions=True)
            
            # 处理结果
            for result in batch_results:
                if isinstance(result, Exception):
                    print(f"[parallel_eval] Task failed with exception: {result}")
                    continue
                all_results.append(result)
            
            pbar.update(len(batch_tasks))
    
    # 整理结果到已经初始化的flags_per_sample中
    # 按sample_idx和step_idx排序
    all_results.sort(key=lambda x: (x.sample_idx, x.step_idx))
    
    for result in all_results:
        # 为非跳过的样本填充结果
        if not flags_per_sample[result.sample_idx]:  # 如果还是空列表
            flags_per_sample[result.sample_idx] = []
        flags_per_sample[result.sample_idx].append(result.is_good)
    
    # 统计信息
    total_time = sum(r.response_time for r in all_results)
    avg_time = total_time / len(all_results) if all_results else 0
    
    stats = {
        "total_tasks": total_tasks,
        "successful_tasks": len(all_results),
        "failed_tasks": total_tasks - len(all_results),
        "total_api_time": total_time,
        "avg_api_time": avg_time,
        "max_concurrent": max_concurrent,
        "fallback_used": False,
        "skipped_samples": skipped_samples,
        "evaluation_type": evaluation_type,
        "model_name": model_name,
        "api_max_retries": api_max_retries
    }
    
    print(f"[parallel_eval] Completed. Stats: {stats}")
    
    # 清理资源
    if api_client:
        await api_client.close()
    
    return flags_per_sample, stats

def _apply_fallback_strategy_parallel(batch) -> List[List[bool]]:
    """并行fallback策略"""
    import random
    
    flags_per_sample = []
    for steps in batch.non_tensor_batch["steps"]:
        flags = [random.choice([True, False]) for _ in steps]
        flags_per_sample.append(flags)
    
    return flags_per_sample

# ————————————————————————————————————————————————————————————————
# 向量化的mask应用（保持不变）
# ————————————————————————————————————————————————————————————————

def apply_step_mask_vectorized(batch,
                             step_flags: List[List[bool]],
                             good_scale: float = 1.0,
                             bad_scale: float = 0.2,
                             neg_bad_scale: float = -0.2,
                             mask_tensor: torch.Tensor = None) -> Dict:
    """
    向量化版本的step mask应用，避免嵌套循环
    对于advantage=0的样本跳过处理
    
    Args:
        batch: 批次数据
        step_flags: step评估结果
        good_scale, bad_scale, neg_bad_scale: 缩放因子
        mask_tensor: 外部传入的mask tensor，shape (bs, resp_len)
                    可以是loss_mask或response_mask，如果为None则使用默认的loss_mask
    
    Returns:
        stats: 应用统计信息
    """
    print(f"[vectorized_mask] Starting vectorized mask application")
    
    # 检查必要的输入
    if 'step_ids' not in batch.batch:
        raise ValueError("batch.batch['step_ids'] is required but not found")
    
    adv = batch.batch["advantages"]  # (bs, resp_len)
    step_ids = batch.batch["step_ids"].to(adv.device)  # (bs, resp_len)
    
    bs, resp_len = adv.shape
    
    if len(step_flags) != bs:
        raise ValueError(f"step_flags length ({len(step_flags)}) != batch size ({bs})")
    
    # 初始化scale为全1
    scale = torch.ones_like(adv)
    
    # 🔧 关键修改：使用外部传入的mask_tensor计算overall advantage
    overall_advs = []
    
    # 使用传入的mask_tensor，如果没有传入则使用默认的loss_mask
    if mask_tensor is not None:
        response_mask = mask_tensor
        print(f"[vectorized_mask] Using external mask tensor with shape {mask_tensor.shape}")
        
        # 验证mask tensor的形状
        if response_mask.shape != (bs, resp_len):
            raise ValueError(f"mask_tensor shape {response_mask.shape} doesn't match expected shape ({bs}, {resp_len})")
    else:
        # 使用默认的loss_mask
        response_mask = batch.batch["loss_mask"][:, -resp_len:]
        print(f"[vectorized_mask] Using default loss_mask")
    
    for sample_idx in range(bs):
        sample_mask = response_mask[sample_idx]
        
        overall_adv = _get_overall_advantage(
            adv[sample_idx], 
            sample_mask
        )
        overall_advs.append(overall_adv)
    
    overall_advs = torch.tensor(overall_advs, device=adv.device)
    overall_pos = overall_advs > 0  # (bs,) bool tensor
    
    # 统计信息
    stats = {
        "total_samples": bs,
        "total_tokens": resp_len * bs,
        "tokens_modified": 0,
        "good_steps": 0,
        "bad_steps": 0,
        "positive_samples": overall_pos.sum().item(),
        "negative_samples": (~overall_pos).sum().item(),
        "zero_adv_samples": 0  # 新增：零advantage样本统计
    }
    
    # 处理每个样本（这部分还是需要循环，但内部是向量化的）
    for b in tqdm(range(bs), desc="[vectorized_mask] Processing samples"):
        current_step_flags = step_flags[b]
        overall_adv_sum = overall_advs[b].item()
        
        # 新增：如果advantage为0，跳过处理（保持scale=1.0）
        if abs(overall_adv_sum) < 1e-8:
            stats["zero_adv_samples"] += 1
            continue
        
        if not current_step_flags:
            continue
            
        # 获取当前样本的step_ids和advantages
        sample_step_ids = step_ids[b]  # (resp_len,)
        sample_adv = adv[b]  # (resp_len,)
        sample_overall_pos = overall_pos[b].item()
        
        # 为每个step创建mask和对应的scale factor
        max_step_id = len(current_step_flags)
        
        # 向量化处理：为每个step_id创建mask
        for step_id, is_good in enumerate(current_step_flags):
            # 创建当前step的token mask
            step_mask = (sample_step_ids == step_id)  # (resp_len,)
            
            if not step_mask.any():
                continue
            
            # 根据overall_pos和is_good确定scale factor
            if sample_overall_pos:
                factor = good_scale if is_good else bad_scale
            else:
                factor = neg_bad_scale if is_good else good_scale
            
            # 应用scale factor
            scale[b].masked_fill_(step_mask, factor)
            
            # 更新统计
            tokens_in_step = step_mask.sum().item()
            stats["tokens_modified"] += tokens_in_step
            
            if is_good:
                stats["good_steps"] += 1
            else:
                stats["bad_steps"] += 1
    
    # 确保填充token（step_id == -1）保持scale=1.0
    padding_mask = (step_ids == -1)
    scale.masked_fill_(padding_mask, 1.0)
    
    # 应用scale
    original_adv_sum = adv.sum().item()
    batch.batch["advantages"] = adv * scale
    new_adv_sum = batch.batch["advantages"].sum().item()
    
    # 保存scale用于调试
    batch.batch["semantic_scale"] = scale
    
    # 更新统计信息
    stats["original_adv_sum"] = original_adv_sum
    stats["new_adv_sum"] = new_adv_sum
    stats["adv_change_ratio"] = new_adv_sum / original_adv_sum if original_adv_sum != 0 else 1.0
    
    print(f"[vectorized_mask] Completed. Advantages: {original_adv_sum:.4f} -> {new_adv_sum:.4f}")
    print(f"[vectorized_mask] Modified {stats['tokens_modified']} tokens ({stats['good_steps']} good steps, {stats['bad_steps']} bad steps)")
    print(f"[vectorized_mask] Skipped {stats['zero_adv_samples']} samples with advantage=0")
    
    return stats

# ————————————————————————————————————————————————————————————————
# 同步包装函数（更新为支持evaluation_type和api_max_retries）
# ————————————————————————————————————————————————————————————————

def evaluate_step_flags(tokenizer,
                        batch,
                        good_words: tuple[str, ...] = ("GOOD",),
                        bad_words: tuple[str, ...] = ("BAD",),
                        model_name: str = "qwen-max",
                        evaluation_type: Literal["local", "api"] = "api",
                        use_parallel: bool = True,
                        max_concurrent: int = 20,
                        mask_tensor: torch.Tensor = None,
                        api_max_retries: int = 200) -> List[List[bool]]:
    """
    兼容性包装函数，可选择使用并行或串行版本，支持本地和API评估
    
    Args:
        tokenizer: 分词器
        batch: 数据批次
        good_words, bad_words: 兼容性参数，在并行版本中未使用
        model_name: 模型名称
        evaluation_type: 评估类型，"local"使用vLLM本地模型，"api"使用API调用
        use_parallel: 是否使用并行版本
        max_concurrent: 最大并发数
        mask_tensor: 外部传入的mask tensor
        api_max_retries: API调用的最大重试次数，特别用于处理429错误
    """
    if use_parallel:
        # 使用异步并行版本
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        flags, stats = loop.run_until_complete(
            evaluate_step_flags_parallel(
                tokenizer=tokenizer,
                batch=batch,
                model_name=model_name,
                evaluation_type=evaluation_type,
                max_concurrent=max_concurrent,
                mask_tensor=mask_tensor,  # 传入外部mask
                api_max_retries=api_max_retries  # 传入API重试次数
            )
        )
        
        print(f"[evaluate_step_flags] Parallel execution stats: {stats}")
        return flags
    else:
        # 使用原来的串行版本（需要从原文件导入）
        print("[evaluate_step_flags] Using serial version (not implemented here)")
        raise NotImplementedError("Serial version not included in parallel implementation")

def apply_step_mask(batch,
                   step_flags: List[List[bool]],
                   good_scale: float = 1.0,
                   bad_scale: float = 0.2,
                   neg_bad_scale: float = -0.2,
                   use_vectorized: bool = True,
                   mask_tensor: torch.Tensor = None):
    """
    兼容性包装函数，可选择使用向量化或原版本
    
    Args:
        batch: 批次数据
        step_flags: step评估结果
        good_scale, bad_scale, neg_bad_scale: 缩放因子
        use_vectorized: 是否使用向量化版本
        mask_tensor: 外部传入的mask tensor
    """
    if use_vectorized:
        stats = apply_step_mask_vectorized(
            batch=batch,
            step_flags=step_flags,
            good_scale=good_scale,
            bad_scale=bad_scale,
            neg_bad_scale=neg_bad_scale,
            mask_tensor=mask_tensor  # 传入外部mask
        )
        return stats
    else:
        # 使用原来的版本（需要从原文件导入）
        print("[apply_step_mask] Using original version (not implemented here)")
        raise NotImplementedError("Original version not included in vectorized implementation")

# ————————————————————————————————————————————————————————————————
# 统一的处理器类（支持evaluation_type和api_max_retries）
# ————————————————————————————————————————————————————————————————

class ParallelSemanticProcessor:
    """并行语义处理器，用于管理整个流程，支持本地和API评估"""
    
    def __init__(self, 
                 max_concurrent: int = 20,
                 batch_size_limit: int = 100,
                 model_name: str = "qwen-max",
                 evaluation_type: Literal["local", "api"] = "api",
                 api_max_retries: int = 200):
        self.max_concurrent = max_concurrent
        self.batch_size_limit = batch_size_limit
        self.model_name = model_name
        self.evaluation_type = evaluation_type
        self.api_max_retries = api_max_retries
        
        # 根据评估类型调整默认参数
        if evaluation_type == "local":
            # 本地推理建议较小的并发数和批次大小
            if max_concurrent > 8:
                print(f"[ParallelSemanticProcessor] Local evaluation: reducing max_concurrent from {max_concurrent} to 8")
                self.max_concurrent = 8
            if batch_size_limit > 32:
                print(f"[ParallelSemanticProcessor] Local evaluation: reducing batch_size_limit from {batch_size_limit} to 32")
                self.batch_size_limit = 32
        
        print(f"[ParallelSemanticProcessor] Initialized with evaluation_type={evaluation_type}")
        print(f"[ParallelSemanticProcessor] Settings: model={model_name}, concurrent={self.max_concurrent}, batch_limit={self.batch_size_limit}, api_retries={self.api_max_retries}")
        
    async def process_batch(self, tokenizer, batch, 
                          good_scale: float = 1.0,
                          bad_scale: float = 0.2,
                          neg_bad_scale: float = -0.2,
                          mask_tensor: torch.Tensor = None) -> Dict:
        """
        处理整个batch的语义评估和mask应用
        对于advantage=0的样本会跳过评估
        
        Args:
            tokenizer: 分词器
            batch: 批次数据
            good_scale, bad_scale, neg_bad_scale: 缩放因子
            mask_tensor: 外部传入的mask tensor，shape (bs, resp_len)
                        可以是loss_mask或response_mask
        
        Returns:
            综合统计信息
        """
        start_time = time.time()
        
        # 1. 并行评估step flags
        eval_method = "vLLM" if self.evaluation_type == "local" else "API"
        print(f"[ParallelSemanticProcessor] Starting step evaluation with {eval_method}...")
        eval_start = time.time()
        
        step_flags, eval_stats = await evaluate_step_flags_parallel(
            tokenizer=tokenizer,
            batch=batch,
            model_name=self.model_name,
            evaluation_type=self.evaluation_type,
            max_concurrent=self.max_concurrent,
            batch_size_limit=self.batch_size_limit,
            mask_tensor=mask_tensor,  # 传入外部mask
            api_max_retries=self.api_max_retries  # 传入API重试次数
        )
        
        eval_time = time.time() - eval_start
        print(f"[ParallelSemanticProcessor] Step evaluation completed in {eval_time:.2f}s")
        
        # 2. 向量化应用mask
        print("[ParallelSemanticProcessor] Applying step mask...")
        mask_start = time.time()
        
        mask_stats = apply_step_mask_vectorized(
            batch=batch,
            step_flags=step_flags,
            good_scale=good_scale,
            bad_scale=bad_scale,
            neg_bad_scale=neg_bad_scale,
            mask_tensor=mask_tensor  # 传入外部mask
        )
        
        mask_time = time.time() - mask_start
        print(f"[ParallelSemanticProcessor] Step mask applied in {mask_time:.2f}s")
        
        # 3. 合并统计信息
        total_time = time.time() - start_time
        
        combined_stats = {
            "total_processing_time": total_time,
            "evaluation_time": eval_time,
            "mask_application_time": mask_time,
            "evaluation_stats": eval_stats,
            "mask_stats": mask_stats,
            "speedup_info": {
                "parallel_evaluation": True,
                "vectorized_masking": True,
                "max_concurrent": self.max_concurrent,
                "evaluation_type": self.evaluation_type,
                "using_vllm": self.evaluation_type == "local",
                "model_name": self.model_name,
                "api_max_retries": self.api_max_retries
            }
        }
        
        print(f"[ParallelSemanticProcessor] Total processing time: {total_time:.2f}s")
        return combined_stats
    
    def process_batch_sync(self, tokenizer, batch, mask_tensor: torch.Tensor = None, **kwargs) -> Dict:
        """
        同步版本的batch处理
        
        Args:
            tokenizer: 分词器
            batch: 批次数据
            mask_tensor: 外部传入的mask tensor
            **kwargs: 其他参数
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.process_batch(tokenizer, batch, mask_tensor=mask_tensor, **kwargs)
        )
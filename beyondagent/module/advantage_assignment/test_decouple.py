from typing import List, Sequence
import torch
from dataclasses import dataclass

@dataclass
class PRMHyper:
    good_step_reward: float = 0.2
    bad_step_reward: float = -0.2
    eps: float = 1e-8

def compute_step_prm_rewards(
    step_flags_list: List[List[bool]], 
    hyper: PRMHyper
) -> List[List[float]]:
    """
    构造基础PRM奖励，只基于步骤质量，不涉及ORM
    """
    prm_rewards_list = []
    
    for flags in step_flags_list:
        rewards = []
        for flag in flags:
            if flag:  # True = good step
                rewards.append(hyper.good_step_reward)
            else:     # False = bad step
                rewards.append(hyper.bad_step_reward)
        prm_rewards_list.append(rewards)
    
    return prm_rewards_list

def normalize_prm_rewards(
    prm_rewards_list: List[List[float]],
    hyper: PRMHyper
) -> List[List[float]]:
    """
    对PRM奖励做批内标准化：(r - mean) / std
    """
    # 拉平所有PRM奖励
    flat_prm = [r for rewards in prm_rewards_list for r in rewards]
    
    if len(flat_prm) == 0:
        return prm_rewards_list
    
    prm_tensor = torch.tensor(flat_prm, dtype=torch.float32)
    prm_mean = prm_tensor.mean()
    prm_std = prm_tensor.std(unbiased=False)
    
    print(f"[PRM标准化] 均值={prm_mean:.4f}, 标准差={prm_std:.4f}")
    
    # 标准化函数
    def normalize_prm(x: float) -> float:
        if prm_std <= hyper.eps:
            return float(x - prm_mean.item())
        else:
            return float((x - prm_mean.item()) / (prm_std.item() + 1e-12))
    
    # 对每个样本的PRM奖励标准化
    norm_prm_list = []
    for rewards in prm_rewards_list:
        norm_rewards = [normalize_prm(r) for r in rewards]
        norm_prm_list.append(norm_rewards)
    
    return norm_prm_list

def normalize_orm_rewards(
    orm_scores: List[float],
    hyper: PRMHyper
) -> List[float]:
    """
    对ORM分数做批内标准化：(score - mean) / std
    """
    if len(orm_scores) == 0:
        return orm_scores
    
    orm_tensor = torch.tensor(orm_scores, dtype=torch.float32)
    orm_mean = orm_tensor.mean()
    orm_std = orm_tensor.std(unbiased=False)
    
    print(f"[ORM标准化] 均值={orm_mean:.4f}, 标准差={orm_std:.4f}")
    
    # 标准化函数
    def normalize_orm(x: float) -> float:
        if orm_std <= hyper.eps:
            return float(x - orm_mean.item())
        else:
            return float((x - orm_mean.item()) / (orm_std.item() + 1e-12))
    
    norm_orm_list = [normalize_orm(score) for score in orm_scores]
    return norm_orm_list

def combine_normalized_rewards(
    norm_prm_list: List[List[float]],
    norm_orm_list: List[float],
    alpha: float = 1.0,
    orm_distribution: str = "last_step"  # "last_step" 或 "all_steps"
) -> List[List[float]]:
    """
    组合标准化的PRM和ORM奖励
    
    Args:
        norm_prm_list: 标准化的PRM奖励
        norm_orm_list: 标准化的ORM奖励  
        alpha: PRM奖励的权重
        orm_distribution: ORM分配方式，"last_step"只加在最后一步，"all_steps"每步都加
    """
    combined_list = []
    
    for i, (norm_prm, norm_orm) in enumerate(zip(norm_prm_list, norm_orm_list)):
        if len(norm_prm) == 0:
            combined_list.append([])
            continue
            
        combined = []
        for j, prm_reward in enumerate(norm_prm):
            if orm_distribution == "last_step":
                # 只在最后一步加ORM
                if j == len(norm_prm) - 1:
                    combined_reward = alpha * prm_reward + norm_orm
                else:
                    combined_reward = alpha * prm_reward
            elif orm_distribution == "all_steps":
                # 每步都加相同的标准化ORM值
                combined_reward = alpha * prm_reward + norm_orm
            else:
                raise ValueError(f"Unknown orm_distribution: {orm_distribution}")
            
            combined.append(combined_reward)
        
        combined_list.append(combined)
        
        # 调试信息
        prm_sum = sum(norm_prm)
        combined_sum = sum(combined)
        if orm_distribution == "last_step":
            expected_sum = alpha * prm_sum + norm_orm
        else:
            expected_sum = alpha * prm_sum + len(norm_prm) * norm_orm
        print(f"[组合] 样本{i}: PRM和={prm_sum:.4f}, 组合和={combined_sum:.4f}, 预期和={expected_sum:.4f}")
    
    return combined_list

def compute_grpo_advantages_from_combined(
    combined_rewards_list: List[List[float]]
) -> List[List[float]]:
    """
    对组合后的奖励做后缀和得到advantage
    """
    advantages_list = []
    
    for rewards in combined_rewards_list:
        if len(rewards) == 0:
            advantages_list.append([])
            continue
            
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        # 后缀和：reverse -> cumsum -> reverse
        advantages = torch.flip(
            torch.cumsum(torch.flip(rewards_tensor, dims=[0]), dim=0), 
            dims=[0]
        ).tolist()
        advantages_list.append(advantages)
    
    return advantages_list

def compute_step_advantages_method4(
    step_flags_list: List[List[bool]],
    orm_scores: List[float],
    hyper: PRMHyper,
    alpha: float = 1.0,
    orm_distribution: str = "last_step"
) -> List[List[float]]:
    """
    方案4主函数：分离标准化PRM和ORM，然后组合做后缀和
    """
    print("=" * 60)
    print("方案4：分离标准化PRM+ORM+后缀和")
    print("=" * 60)
    
    # 1. 构造基础PRM奖励
    prm_rewards = compute_step_prm_rewards(step_flags_list, hyper)
    
    # 2. 分别标准化PRM和ORM
    norm_prm = normalize_prm_rewards(prm_rewards, hyper)
    norm_orm = normalize_orm_rewards(orm_scores, hyper)
    
    # 3. 组合标准化奖励
    combined_rewards = combine_normalized_rewards(norm_prm, norm_orm, alpha, orm_distribution)

    # 4. GRPO后缀和
    advantages = compute_grpo_advantages_from_combined(combined_rewards)
    
    return advantages

def check_order_per_sample(flags: List[bool], rewards: List[float], advantages: List[float], sample_idx: int, orm_score: float):
    """检查单个样本的结果是否合理"""
    print(f"  样本{sample_idx} (ORM={orm_score:.3f}):")
    print(f"    Flags: {flags}")
    print(f"    Rewards: {[f'{x:.4f}' for x in rewards]}")
    print(f"    Advantages: {[f'{x:.4f}' for x in advantages]}")
    
    # 检查good步骤和bad步骤的advantage趋势
    good_advs = [advantages[i] for i in range(len(flags)) if flags[i]]
    bad_advs = [advantages[i] for i in range(len(flags)) if not flags[i]]
    
    if good_advs and bad_advs:
        good_mean = sum(good_advs) / len(good_advs)
        bad_mean = sum(bad_advs) / len(bad_advs)
        print(f"    GOOD步骤平均advantage={good_mean:.4f}, BAD步骤平均advantage={bad_mean:.4f}")

        if good_mean <= bad_mean:
            print(f"    ⚠️  轨迹中GOOD步骤advantage不应低于BAD步骤")
        
def test_method4():
    # print("方案4测试：分离标准化PRM+ORM")
    
    hyper = PRMHyper()
    alpha = 0.2 # PRM权重
    
    # 测试1: 成功轨迹批次，多数good步骤
    print("\n【测试1: 成功轨迹批次，多数GOOD步骤】")
    orms1 = [1.0, 1.0]
    flags1 = [
        [True, True, False, True, True, False, True, True, True],  # 7G,2B
        [True, True, True, False, True, True, False]               # 5G,2B
    ]
    advantages1 = compute_step_advantages_method4(flags1, orms1, hyper, alpha, "last_step")
    
    # 获取组合后的奖励用于分析
    prm1 = compute_step_prm_rewards(flags1, hyper)
    norm_prm1 = normalize_prm_rewards(prm1, hyper)
    norm_orm1 = normalize_orm_rewards(orms1, hyper)
    combined1 = combine_normalized_rewards(norm_prm1, norm_orm1, alpha, "last_step")
    
    for i in range(len(orms1)):
        check_order_per_sample(flags1[i], combined1[i], advantages1[i], i, orms1[i])

    # 测试2: 失败轨迹批次，多数bad步骤
    print("\n【测试2: 失败轨迹批次，多数BAD步骤】")
    orms2 = [-1.0, -1.0]
    flags2 = [
        [True, False, False, True, False, False, False, False, False],  # 2G,7B
        [False, True, False, False, False, False]                       # 1G,5B
    ]
    
    advantages2 = compute_step_advantages_method4(flags2, orms2, hyper, alpha, "last_step")
    
    # 获取组合后的奖励用于分析
    prm2 = compute_step_prm_rewards(flags2, hyper)
    norm_prm2 = normalize_prm_rewards(prm2, hyper)
    norm_orm2 = normalize_orm_rewards(orms2, hyper)
    combined2 = combine_normalized_rewards(norm_prm2, norm_orm2, alpha, "last_step")
    
    for i in range(len(orms2)):
        check_order_per_sample(flags2[i], combined2[i], advantages2[i], i, orms2[i])

    # 测试3: 混合批次
    print("\n【测试3: 混合批次】")
    orms3 = [1.0, -1.0]
    flags3 = [
        [True, True, False, True, True],      # 成功: 4G,1B
        [False, False, True, False, False]    # 失败: 1G,4B
    ]
    
    advantages3 = compute_step_advantages_method4(flags3, orms3, hyper, alpha, "last_step")
    
    # 获取组合后的奖励用于分析
    prm3 = compute_step_prm_rewards(flags3, hyper)
    norm_prm3 = normalize_prm_rewards(prm3, hyper)
    norm_orm3 = normalize_orm_rewards(orms3, hyper)
    combined3 = combine_normalized_rewards(norm_prm3, norm_orm3, alpha, "last_step")
    
    for i in range(len(orms3)):
        check_order_per_sample(flags3[i], combined3[i], advantages3[i], i, orms3[i])

    # 测试4: 对比不同ORM分配方式
    print("\n【测试4: 对比ORM分配方式】")
    print("4a. ORM只加在最后一步：")
    advantages4a = compute_step_advantages_method4(flags3, orms3, hyper, alpha, "last_step")
    
    print("\n4b. ORM每步都加：")
    advantages4b = compute_step_advantages_method4(flags3, orms3, hyper, alpha, "all_steps")
    
    # 简单对比
    print("\n对比结果:")
    for i in range(len(orms3)):
        print(f"  样本{i} last_step模式: {[f'{x:.4f}' for x in advantages4a[i]]}")
        print(f"  样本{i} all_steps模式: {[f'{x:.4f}' for x in advantages4b[i]]}")

    # 测试5: 边界情况
    print("\n【测试5: 边界情况】")
    print("5a. 全GOOD：")
    orms5a = [1.0, 1.0]
    flags5a = [[True, True, True], [True, True, True, True]]
    advantages5a = compute_step_advantages_method4(flags5a, orms5a, hyper, alpha, "last_step")
    
    print("5b. 全BAD：")
    orms5b = [-1.0, -1.0]
    flags5b = [[False, False, False], [False, False, False, False]]
    advantages5b = compute_step_advantages_method4(flags5b, orms5b, hyper, alpha, "last_step")
    
    print("5c. ORM分数不同：")
    orms5c = [1.0, -1.0]
    flags5c = [[True, False, True], [False, True, False]]
    advantages5c = compute_step_advantages_method4(flags5c, orms5c, hyper, alpha, "last_step")

if __name__ == "__main__":
    test_method4()
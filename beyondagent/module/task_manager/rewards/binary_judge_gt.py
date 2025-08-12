import re
import threading
from typing import Any, Optional, cast
from loguru import logger
from beyondagent.client.env_client import EnvClient
from beyondagent.client.llm_client import DashScopeClient
from beyondagent.module.agent_flow.reward_calculator import RewardCalculator
from beyondagent.schema.task import Task
from beyondagent.schema.trajectory import Trajectory

from . import grader_manager

USER_PROMPT = """
### Role Description
You are an expert AI agent evaluator. Your task is to assess an agent's performance based on its action trajectory and the original user request. Apply strict binary scoring (0/1) for each dimension without partial credits.

### Input Analysis
You will receive two inputs:
1. User Task: The original objective the agent should accomplish
2. Agent Trajectory: Sequential record of actions, decisions, and outputs during task execution

### Evaluation Procedure
Follow these steps sequentially:

#### Step 1: Critical Failure Check
Immediately score both dimensions 0 if ANY of these occur:
- Outputs gibberish/unreadable content
- Enters infinite loop or identical step repetition
- Completely irrelevant to user task
- Fails to produce any actionable output

#### Step 2: Task Intent Comprehension (Score 0 or 1)
- Score 1 ONLY if: 
  Agent accurately identifies core objective
  Initial actions align with task purpose
- Score 0 if:
  Misinterprets fundamental task purpose
  Shows contradictory understanding

#### Step 3: Task Correct Completion (Score 0 or 1)
Score 1 ONLY if ALL conditions are met:
- Every step is logically valid and necessary, or is recovered fastly in subsequent steps
- Zero hallucinated information
- Final output fully resolves user's request
- All intermediate steps are correct
Score 0 if ANY failure occurs

To evaluate the steps, we provide you with a reference solution to compare against. Please note that this solution demonstrates a correct approach and outcome, but it may not be the *only* correct way to solve the task. A different but equally valid solution should also be considered successful.

{reference_trajs}

### Mandatory Constraints
- Never combine scores or calculate totals
- Critical failure overrides all other checks
- Scores are independent (e.g., may score 1,0)

### Output
**Strictly follow this sequence:**
1. Perform Step 1 → Step 2 → Step 3 evaluations in order
2. Generate analysis covering all evaluation steps
3. Finaly output the evaluation result with the following FORMAT:
Reason: [Reason for score]
Critical Failure: [Yes/No]  
Intent Comprehension: [0/1]
Correct Completion: [0/1]

** User Task **:
{task}

** Agent Trajectory (STEP-ACTION-OBSERVATION) **:
{trajs}

** Reminder **:
Perform evaluation steps sequentially before generating output.
Over the past period of time, the average score you gave to some samples was {running_mean:.4f}.
Please note that the average score must be maintained around {mean_score:.4f} (+-0.2), or you will be penalized.
"""

def steps_to_msg(steps: list[dict[str, Any]]) -> str:
    # 添加轨迹消息（将所有对话转换为一个连贯的文本）
    trajectory_text = ""
    assert steps[0]['role'] == 'assistant'
    for i, msg in enumerate(steps):
        role = msg.get("role", "unknown")
        if role == 'assistant':
            block = f""">>> STEP {i} <<<
<|ACTION|>
{msg['content']}
<|END|>
"""
        elif role == "user":
            block = f"""<|OBSERVATION|>
{msg['content']}
<|END|>
"""
        else:
            raise ValueError("roles in trajectory must be assistant or user")
        trajectory_text += block.strip() + "\n\n"
    return trajectory_text

@grader_manager.reg("llm-binary-gt")
class LlmAsJudgeBinaryRewardCalculatorWithGT(RewardCalculator):
    """
    RewardCalculator that uses LLM as judge.
    """
    # 定义类变量，跨实例共享
    _running_judge_mean = 0.5  # 初始化为默认值
    _update_lock = threading.Lock()  # 锁也需要作为类变量共享

    def __init__(self, task: Task, model_name='qwq-plus', mean_score: float = 0.5):
        super().__init__(task)

        self._client = DashScopeClient(model_name=model_name)
        self._mean_score = mean_score

    @classmethod
    def update_running_judge_mean(cls, new_score: float):
        """
        更新类变量 `_running_judge_mean`，用锁来保证线程安全。
        """
        with cls._update_lock:
            cls._running_judge_mean = 0.9 * cls._running_judge_mean + 0.1 * new_score

    @classmethod
    def get_running_judge_mean(cls):
        """
        获取当前的 `_running_judge_mean`。
        """
        with cls._update_lock:
            return cls._running_judge_mean
    
    def pack_message(self, trajectory: Trajectory):
        """Pack trajectory into a message.
        
        Args:
            trajectory (Trajectory): trajectory to pack
        """
        messages = []
        
        assert len(trajectory.steps) >= 2 and trajectory.steps[1]['role'] == 'user', "trajectory must start with system message and then user message"
        task_query = trajectory.steps[1]['content']
        
        # TODO 至少现在我们的合成任务 gt 一定不是空的
        assert self.task.ground_truth is not None, "ground truth must not be None for synthetic task"
        messages.append({
            "role": "user",
            "content": USER_PROMPT.format(
                task=task_query, 
                trajs=steps_to_msg(trajectory.steps[2:]),
                running_mean=self.get_running_judge_mean(),
                mean_score=self._mean_score, reference_trajs=self.task.ground_truth or "[No solution provided, please judge the task by yourself]"
                )
            }
        )
        return messages

    
    def calculate_reward(self, trajectory: Trajectory, env: EnvClient, instance_id: str) -> float:
        x = cast(float, self._calculate_reward(trajectory, env, eject_llm_output=False))
        return x
        

    def _calculate_reward(self, trajectory: Trajectory, env: EnvClient, *, eject_llm_output: bool = False):
        """Calculate reward for a trajectory in specific environment.
        
        Args:
            trajectory (Trajectory): trajectory to calculate reward
            env (EnvClient): environment where the trajectory is executed
        """
        response = ""
        for chunk in self._client.chat_stream_with_retry(messages=self.pack_message(trajectory), max_retries=64):
            response += chunk
        
        # 默认分数
        score: float = 0.0

        if response:
            try:
                # 解析结果，兼容大小写与多余空格
                cf_match = re.search(r"Critical\s*Failure\s*:\s*(Yes|No)\b", response, re.IGNORECASE)
                intent_match = re.search(r"Intent\s*Comprehension\s*:\s*([01])\b", response, re.IGNORECASE)
                correct_match = re.search(r"Correct\s*Completion\s*:\s*([01])\b", response, re.IGNORECASE)

                critical = bool(cf_match and cf_match.group(1).strip().lower().startswith("y"))
                intent_score = int(intent_match.group(1)) if intent_match else 0
                correct_score = int(correct_match.group(1)) if correct_match else 0

                if critical:
                    score = 0.0
                else:
                    score = 0.2 * intent_score + 0.8 * correct_score
                    score = correct_score

                logger.info(
                    f"LLM judge parsed -> critical={critical}, intent={intent_score}, correct={correct_score}, reward={score}"
                )
            except Exception as e:
                logger.exception(f"Failed to parse LLM judge response: {e}. Raw response: {response!r}")
                score = 0.0
        else:
            logger.warning("Empty LLM judge response; setting score=0.0")
        
        self.update_running_judge_mean(score)

        if not eject_llm_output:
            return score
        else:
            return score, response
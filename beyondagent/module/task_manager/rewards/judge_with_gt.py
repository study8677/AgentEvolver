from typing import Optional, cast
from beyondagent.client.env_client import EnvClient
from beyondagent.client.llm_client import DashScopeClient
from beyondagent.module.agent_flow.reward_calculator import RewardCalculator
from beyondagent.schema.task import Task
from beyondagent.schema.trajectory import Trajectory

from . import grader_manager

USER_PROMPT="""Based on the conversation trajectory above and the provided **Reference Solution**, evaluate the task completion quality using the framework provided.

---
To assist in your evaluation, a reference solution is provided below. Please note that this solution demonstrates a correct approach and outcome, but it may not be the *only* correct way to solve the task. A different but equally valid solution from the agent should also be considered successful.

{{reference_solution}}
---

Your evaluation should address the following dimensions in order:

**Step 1: Relevance Check (0 or proceed)**
- Are the solution steps relevant to the problem? If the approach is completely unrelated to the task requirements, assign 0 points immediately.
- If relevant, proceed to other evaluation dimensions.

**Step 2: Repetition Penalty Check**
- Does the agent get stuck in infinite loops or repeat identical steps endlessly?
- If there are infinite repetitions of the same steps, consider the relevance of existing steps:
 - If steps are relevant: Maximum 20 points
 - If steps are irrelevant: 0 points

**Step 3: Goal Achievement Assessment (Critical Binary Check)**
- Examine ALL steps comprehensively to determine if the task goal is truly achieved.
- **Crucially, compare the agent's final result and solution path with the provided "Reference Solution" to aid in judging correctness.**
- Do not be misled by superficial language - verify actual completion.
- If the agent's solution is different from the reference but is also correct and achieves the same goal, it MUST be considered as the goal being achieved.

**MANDATORY SCORING CONSTRAINTS:**
- If steps are relevant AND goal is achieved/answer is correct: Score MUST be 60-100
- If steps are relevant BUT goal is not achieved/answer is incorrect: Score MUST be 0-40
- FORBIDDEN: Do not assign scores between 41-59

**Step 4: Additional Deductions (within the above constraints)**
- **Code Execution Errors**: Deduct points for runtime errors, bugs, or failed executions.
- **Efficiency and Conciseness**: Compare the agent's steps to the "Reference Solution". If the agent's approach is significantly more cumbersome, redundant, or roundabout than the reference, deduct points accordingly, even if the final answer is correct. Unnecessary or irrelevant steps fall under this category.

**Scoring Guidelines:**
- 90-100: Exceptional performance - goal achieved with an efficient and clean execution, comparable to or better than the reference solution.
- 80-89: Strong performance - goal achieved with minor inefficiencies or small errors when compared to the reference.
- 70-79: Good performance - goal achieved, but the process was notably less efficient or contained more unnecessary steps than the reference.
- 60-69: Adequate performance - goal achieved but with significant problems in efficiency, clarity, or execution.
- 30-40: Poor performance - goal not achieved but relevant approach with some progress, showing some understanding of the path outlined in the reference.
- 10-29: Very poor performance - goal not achieved with major execution issues or a path that barely aligns with a correct solution.
- 1-9: Minimal relevant attempt - goal not achieved with severe problems.
- 0: Complete failure - irrelevant approach or infinite repetition of irrelevant steps.

**REMEMBER**:
- No scores between 41-59 are allowed.
- Goal achievement determines the 60+ vs 0-40 range.
- The Reference Solution is a guide for correctness and efficiency, not a rigid script.

Provide your detailed analysis first, explaining your reasoning for each evaluation dimension. Then assign a precise integer score following the mandatory constraints above.

First provide your detailed reasoning analysis, then output an integer score between 0-40 or 60-100 enclosed in `<reward></reward>` tags, e.g., `<reward>75</reward>`
"""

@grader_manager.reg("llm-gt")
class LlmAsJudgeRewardCalculatorWithGT(RewardCalculator):
    """
    RewardCalculator that uses LLM as judge.
    
    TODO: This is a temperary solution for synthetic data.
    """
    def __init__(self, task:Task, model_name='qwq-plus'):
        super().__init__(task)
        self._client=DashScopeClient(model_name=model_name)
    
    def pack_message(self, trajectory: Trajectory):
        """Pack trajectory into a message.
        
        Args:
            trajectory (Trajectory): trajectory to pack
        """
        messages=[]
        
        # 添加轨迹消息（将所有对话转换为一个连贯的文本）
        trajectory_text = "The following is the dialogue trace of the task execution:\n\n"
        for i, msg in enumerate(trajectory.steps):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            trajectory_text += f"{role.upper()}: {content}\n\n"
        
        messages.append({"role": "user", "content": trajectory_text})
        user_prompt=USER_PROMPT.replace("{{reference_solution}}",self.task.ground_truth or "[No solution provided, please judge the task by yourself]")
        messages.append({"role":"user","content":user_prompt})
        return messages
    
    
    def calculate_reward(self, trajectory: Trajectory, env: EnvClient, instance_id: str) -> float:
        x=cast(float,self._calculate_reward(trajectory,env,eject_llm_output=False))
        return x
        

    def _calculate_reward(self, trajectory: Trajectory, env:EnvClient, *, eject_llm_output:bool=False):
        """Calculate reward for a trajectory in specific environment.
        
        Args:
            trajectory (Trajectory): trajectory to calculate reward
            env (EnvClient): environment where the trajectory is executed
        """
        response=""
        for chunk in self._client.chat_stream_with_retry(messages=self.pack_message(trajectory),max_retries=64):
            response += chunk
        if response:
            import re
            reward_match = re.search(r'<reward>([\d\.]+)</reward>', response.strip())
            if reward_match:
                score = float(reward_match.group(1))
                score = max(0.0, min(100.0, score))/100.0
            else:
                print(f"Could not parse score from response: {response}")
                score=0.0
        else:
            print("No response from evaluation API")
            score=0.0
        
        if not eject_llm_output:
            return score
        else:
            return score,response
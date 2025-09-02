from beyondagent.client.env_client import EnvClient
from beyondagent.client.llm_client import DashScopeClient
from beyondagent.module.agent_flow.reward_calculator import RewardCalculator
from beyondagent.schema.trajectory import Trajectory

USER_PROMPT="""Based on the conversation trajectory above, evaluate the task completion quality using the framework provided.

Your evaluation should address:

**Task Understanding**: Does the agent correctly interpret the requirements and objectives?
**Strategic Planning**: Is the approach logical, efficient, and well-structured?
**Execution Quality**: Are the actions appropriate, accurate, and effectively implemented?
**Completion Level**: To what extent are the task goals achieved?
**Error Recovery**: How well does the agent handle mistakes and adapt?

Provide your detailed analysis first, explaining your reasoning for each evaluation dimension. Then assign a precise continuous score between 0.0 and 1.0, where:
- 0.9-1.0: Exceptional performance with complete success
- 0.7-0.9: Strong performance with minor issues
- 0.5-0.7: Adequate performance with notable gaps
- 0.3-0.5: Poor performance with major deficiencies  
- 0.1-0.3: Very poor performance with minimal progress
- 0.0-0.1: Complete failure or no meaningful attempt

First provide your detailed reasoning analysis, then output a continuous score between 0.0 and 1.0 enclosed in <reward></reward> tags, e.g., <reward>0.75</reward>
"""

class LlmAsJudgeRewardCalculator(RewardCalculator):
    """
    RewardCalculator that uses LLM as judge.
    
    TODO: This is a temperary solution for synthetic data.
    """
    def __init__(self):
        self._client=DashScopeClient()
    
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
        messages.append({"role":"user","content":USER_PROMPT})
        return messages
        

    def calculate_reward(self, trajectory: Trajectory, env:EnvClient) -> float:
        """Calculate reward for a trajectory in specific environment.
        
        Args:
            trajectory (Trajectory): trajectory to calculate reward
            env (EnvClient): environment where the trajectory is executed
        """
        response=self._client.chat_with_retry(messages=self.pack_message(trajectory))
        if response:
            import re
            reward_match = re.search(r'<reward>([\d\.]+)</reward>', response.strip())
            if reward_match:
                score = float(reward_match.group(1))
                score = max(0.0, min(1.0, score))
                return score
            else:
                print(f"Could not parse score from response: {response}")
                return 0.0  # 默认分数
        else:
            print("No response from evaluation API")
            return 0.0  # 默认分数
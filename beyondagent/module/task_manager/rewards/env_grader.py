from typing import cast
from beyondagent.client.env_client import EnvClient
from beyondagent.client.llm_client import DashScopeClient
from beyondagent.module.agent_flow.reward_calculator import RewardCalculator
from beyondagent.schema.task import Task
from beyondagent.schema.trajectory import Trajectory

from . import grader_manager

@grader_manager.reg("env")
class EnvGrader(RewardCalculator):
    def __init__(self, task:Task):
        super().__init__(task)
        pass
    
    def calculate_reward(self, trajectory: Trajectory, env: EnvClient, instance_id: str) -> float:
        score = env.evaluate(instance_id, params={"sparse": True})
        return score
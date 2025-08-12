from abc import ABC, abstractmethod

from beyondagent.client.env_client import EnvClient
from beyondagent.schema.task import Task
from beyondagent.schema.trajectory import Trajectory

class RewardCalculator(ABC):
    def __init__(self,task: Task):
        self._task=task
    
    @property
    def task(self):
        return self._task
        
    @abstractmethod
    def calculate_reward(self, trajectory: Trajectory, env:EnvClient, instance_id:str) -> float:
        """Calculate reward for a trajectory in specific environment.
        
        Args:
            trajectory (Trajectory): trajectory to calculate reward
            env (EnvClient): environment where the trajectory is executed
        """
        pass
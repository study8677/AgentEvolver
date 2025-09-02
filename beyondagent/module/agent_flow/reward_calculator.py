from abc import ABC, abstractmethod

from beyondagent.client.env_client import EnvClient
from beyondagent.schema.trajectory import Trajectory

class RewardCalculator(ABC):
    @abstractmethod
    def calculate_reward(self, trajectory: Trajectory, env:EnvClient) -> float:
        """Calculate reward for a trajectory in specific environment.
        
        Args:
            trajectory (Trajectory): trajectory to calculate reward
            env (EnvClient): environment where the trajectory is executed
        """
        pass
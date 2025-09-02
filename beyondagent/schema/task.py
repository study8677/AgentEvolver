from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class Task(BaseModel):
    """
    Task used in training.
    """
    task_id: str = Field(default=...)

    env_type: str = Field(default="appworld")

    metadata: dict = Field(default_factory=dict)

    query: Optional[str] = Field(default=None) # query. if set, the original query will be replaced by this before training.
    
    evaluator: str = Field(default="env")


class TaskObjective(BaseModel):
    """
    Task used in task exploration and extraction only.
    
    It includes the custom ground_truth and reward, which may be merged to Task and used in training in future.
    """
    
    task:Task=Field(...,description="task")
    ground_truth:str=Field(...,description="ground truth")
    confidence:Optional[float]=Field(None,description="confidence")
    reward:Optional[float]=Field(None,description="reward")
    
    @property
    def objective(self):
        return self.task.query
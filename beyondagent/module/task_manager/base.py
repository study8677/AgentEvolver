import abc
from typing import Any, Protocol

from beyondagent.schema.task import Task, TaskObjective


class LlmClient(Protocol):
    def chat(
        self, messages: list[dict[str, str]], sampling_params: dict[str, Any]
    ) -> str: ...


class TaskObjectiveRetrieval(abc.ABC):
    """支持任务相关任务 objective 检索，用于避免重复探索"""

    @abc.abstractmethod
    def retrieve_objectives(self, task: Task) -> list[TaskObjective]: ...

    @abc.abstractmethod
    def add_objective(self, objective: TaskObjective): ...
    
    @abc.abstractmethod
    def reset(self):...



class NaiveTaskObjectiveRetrieval(TaskObjectiveRetrieval):

    def __init__(self):
        # 目前单次训练中只会有同一个 env_type 的 task，所以可以直接使用 task_id as key
        self._mp: dict[str, list[TaskObjective]] = {}

    def retrieve_objectives(self, task: Task) -> list[TaskObjective]:
        if task.task_id not in self._mp:
            return []
        return self._mp[task.task_id]

    def add_objective(self, objective: TaskObjective):
        if objective.task.task_id not in self._mp:
            self._mp[objective.task.task_id] = []

        self._mp[objective.task.task_id].append(objective)
    
    def reset(self):
        self._mp = {}
import abc
from typing import Sequence

from beyondagent.schema.task import TaskObjective


class TaskPostFilter(abc.ABC):
    @abc.abstractmethod
    def filter(self, tasks: Sequence[TaskObjective]) -> list[TaskObjective]:
        pass


class NaiveTaskPostFilter(TaskPostFilter):
    def filter(self, tasks: Sequence[TaskObjective]) -> list[TaskObjective]:
        tasks = list(tasks)
        tasks.sort(key=lambda x: x.confidence or 0, reverse=True)

        # 简单去重：基于查询文本相似性
        unique_tasks = []
        seen_queries = set()

        for i, task in enumerate(tasks):
            # 简化查询用于去重比较
            query = task.objective
            assert query is not None
            normalized_query = (
                query.lower().strip()
            )  # FIXME: this only supports English

            # 检查是否已存在相似查询
            is_duplicate = False
            for seen_query in seen_queries:
                if self._check_similarity(normalized_query, seen_query):
                    is_duplicate = True
                    break

            if task.ground_truth != "" and not is_duplicate:
                unique_tasks.append(task)
                seen_queries.add(normalized_query)

        return unique_tasks

    def _check_similarity(
        self, query1: str, query2: str, threshold: float = 0.8
    ) -> bool:
        """简单的查询相似性检查"""
        # 基于词汇重叠的简单相似性度量
        words1 = set(query1.split())
        words2 = set(query2.split())

        if not words1 or not words2:
            return False

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        similarity = len(intersection) / len(union) if union else 0
        return similarity >= threshold

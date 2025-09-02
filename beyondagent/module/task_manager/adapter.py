import json
import tempfile
from typing import Iterator, Optional, Sequence
import uuid


from beyondagent.schema.task import Task, TaskObjective
from verl.utils.dataset.rl_dataset import RLHFDataset
from torch.utils.data import IterableDataset
import pandas as pd
from omegaconf import DictConfig, ListConfig
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.processing_utils import ProcessorMixin

def convert_to_tasks(dataset:RLHFDataset,env_type:str)->list[Task]:
    """将来自环境的原本 RLHFDataset 转为供 TaskManager 使用的 Task 列表
    """
    res=[]
    for record in dataset:
        # set query to None to disable query replacement
        task = Task(
            task_id=record["extras"]["task_id"],
            env_type=env_type,
            evaluator="env",
        )
        res.append(task)
    
    return res

def to_rl_dataset(
    tasks: Sequence[TaskObjective],
    tokenizer: PreTrainedTokenizer,
    config: DictConfig,
    processor: Optional[ProcessorMixin] = None,
) -> RLHFDataset:
    processed_records = []

    for id,task_obj in enumerate(tasks):
        task = task_obj.task

        # 构建 reward_model
        ground_truth = [task_obj.ground_truth] if task_obj.ground_truth else []

        # 构建单条记录
        record = {
            "data_source": task.env_type,
            "prompt": [{"content": str(task.task_id), "role": "user"}], # `prompt` is never used. trainer will get trajectories from env. metrics code needs this to group results.
            "reward_model": {"ground_truth": ground_truth, "style": "rule"},
            "uuid": str(uuid.uuid4()),
            "extras": {
                "task_id": task.task_id,
                "new_query": task.query,
                "evaluator": task.evaluator,
            },
        }

        processed_records.append(record)

    df = pd.DataFrame(processed_records)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        df.to_parquet(f.name)

    # 转换为 Dataset
    return RLHFDataset([f.name], tokenizer, config, processor)


class OnflyRlDataset(IterableDataset):
    def __init__(self, release_used_dataset: bool = True):
        super().__init__()
        self._do_release_used_dataset = release_used_dataset

        self._datasets: list[RLHFDataset] = []
        self._passed_datasets_cnt = 0
        self._cur_dataset = 0
        self._cur = 0

    def __len__(self):
        pass

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        if len(self._datasets) <= self._cur_dataset:
            raise StopIteration

        this_cur = self._cur - self._passed_datasets_cnt
        if this_cur >= len(self._datasets[self._cur_dataset]):
            self._passed_datasets_cnt += len(self._datasets[self._cur_dataset])
            self._cur_dataset += 1
            this_cur = 0

        if len(self._datasets) <= self._cur_dataset:
            raise StopIteration

        # release used datasets
        if self._do_release_used_dataset:
            self._release_used_dataset()

        self._cur += 1
        return self._datasets[self._cur_dataset][this_cur]

    @property
    def num_rest_data(self) -> int:
        return sum([len(d) for d in self._datasets[self._cur_dataset :]]) - (
            self._cur - self._passed_datasets_cnt
        )

    def append_dataset(self, dataset: RLHFDataset):
        self._datasets.append(dataset)

    def _release_used_dataset(self):
        self._datasets = self._datasets[self._cur_dataset :]
        self._cur_dataset = 0

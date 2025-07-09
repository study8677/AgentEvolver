from concurrent.futures import ThreadPoolExecutor
import copy
import functools
import json
import os
import threading
import time
from typing import (
    Callable,
    Iterable,
    NotRequired,
    Optional,
    Sequence,
    TypedDict,
    Unpack,
)

import hydra
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import IterableDataset
from tqdm import tqdm
from beyondagent.client.llm_client import DashScopeClient
from beyondagent.module.agent_flow.agent_flow import AgentFlow
from beyondagent.module.agent_flow.base_agent_flow import BaseAgentFlow
from beyondagent.module.task_manager import adapter
from beyondagent.module.task_manager.adapter import OnflyRlDataset, to_rl_dataset
from beyondagent.module.task_manager.explorer import Explorer
from beyondagent.module.task_manager.filters import TaskPostFilter
from beyondagent.module.task_manager.prompts.prompt_explore import (
    get_agent_interaction_system_prompt,
)
from beyondagent.module.task_manager.prompts.prompt_summarize import (
    get_task_summarize_prompt,
    parse_tasks_from_response,
)
from beyondagent.module.task_manager.protocols import LlmClient, TaskObjectiveRetrieval
from beyondagent.schema.task import Task, TaskObjective
from beyondagent.schema.trajectory import Trajectory
from verl.utils.dataset.rl_dataset import RLHFDataset


class TaskManagerProps(TypedDict):
    max_llm_retries: int
    max_explore_step: int
    num_explore_threads: int
    n: int
    exploration_llm_temperature: NotRequired[float]
    exploration_llm_top_p: NotRequired[float]
    exploration_llm_top_k: NotRequired[int]
    task_summary_history_length: NotRequired[int]


# TODO: 针对不同环境的统一接口，message-in message-out？那可能不需要这个
# TODO: 能够替换的 exploration & extraction (summary) strategy


class TaskManager(object):

    def __init__(
        self,
        config: DictConfig,
        llm_client: LlmClient,
        old_retrival: TaskObjectiveRetrieval,
        tokenizer,
        env_service_url: str,
        **kwargs: Unpack[TaskManagerProps],
    ):
        self._config = config
        self._llm_client = llm_client
        self._old_retrival = old_retrival
        self._env_service_url = env_service_url
        self._tokenizer = tokenizer  # cc: 这玩意似乎不该在这
        self._max_llm_retries = kwargs["max_llm_retries"] or 3
        self._max_explore_step = kwargs["max_explore_step"] or 10
        self._num_exploration_threads = kwargs["num_explore_threads"] or 10
        self._n = kwargs["n"]
        self._exploration_llm_temperature = kwargs.get(
            "exploration_llm_temperature", 1.0
        )
        self._exploration_llm_top_p = kwargs.get("exploration_llm_top_p", 1.0)
        self._exploration_llm_top_k = kwargs.get("exploration_llm_top_k", 1)
        self._task_summary_history_length = kwargs.get("task_summary_history_length", self._max_explore_step)

        self._filters: list[TaskPostFilter] = []

    def register_filter(self, filter: TaskPostFilter):
        self._filters.append(filter)

    def generate_task(self, tasks: Sequence[Task],*,show_progress=False) -> list[TaskObjective]:
        task_q = list(copy.copy(tasks)) * self._n
        res = []
        # 每次最多探索所有不同任务，或者最大线程个任务，防止同批次中生成相同任务
        parallel_num = min(self._num_exploration_threads, len(tasks))
        for i in tqdm(range(0, len(task_q), parallel_num), disable=not show_progress):
            trajectories = self._step_explore_batch(task_q[i : i + parallel_num])
            task_objectives = self._step_summarize_batch(
                task_q[i : i + parallel_num], trajectories
            )
            res.extend(task_objectives)

        # post filter
        res = functools.reduce(lambda x, f: f.filter(x), self._filters, res)

        return res

    def get_dataset(self, tasks: Iterable[Task], bs: int, tokenizer, config):
        """
        Get dataset.

        Args:
            tasks: Iterable[Task]
            bs: int. 该 batch size 决定一次读取的 task 数量。每次生成的 dataset 大小为 bs * self._n。
            tokenizer: transformers.tokenization_utils.PreTrainedTokenizer
            config: DictConfig. Only for RLHFDataset.
        """
        fa = self
        
        lock=threading.Lock()

        # wrapper for data auto-reloading
        class AutoReloadDataset(IterableDataset):
            def __init__(self, bs: int):
                self._bs = bs

                self._dataset = OnflyRlDataset(release_used_dataset=True)
            
            def reload(self):
                # avoid data loader calling reload multiple times
                with lock:
                    logger.debug('reloading...') # this should only happen once
                    # avoid data loader calling reload multiple times
                    if self._dataset.num_rest_data > 0:
                        return self._dataset.num_rest_data

                    delta = []
                    for task in tasks:
                        delta.append(task)
                        if len(delta) == self._bs:
                            break

                    ls = fa.generate_task(delta)
                    while len(ls) < self._bs * fa._n:
                        logger.debug("failed to generate enough tasks, retrying")
                        ls = fa.generate_task(delta)

                    self._dataset.append_dataset(to_rl_dataset(ls, tokenizer, config))
                    return self._dataset.num_rest_data

            def __iter__(self):
                return self

            def __next__(self):
                if self._dataset.num_rest_data == 0:
                    logger.debug("no data left")
                    if self.reload() == 0:
                        logger.debug("no task left, stop reloading and iteration")
                        raise StopIteration
                return next(self._dataset)

        return AutoReloadDataset(bs)
    
    def load_persistent_dataset(self,tasks:Sequence[Task],filepath:str,*,config,tokenizer,processor)->RLHFDataset:
        """保持任务探索结果不变的数据集。探索一次后保存到文件，后续再加载。
        """
        if not os.path.exists(filepath):
            logger.info("no persistent file, exploring tasks. this will take a while...")
            objectives=self.generate_task(tasks[:1],show_progress=True) # FIXME: debug
            with open(filepath,"w") as f:
                    f.writelines([ob.json() for ob in objectives])
        else:
            logger.info("loading persistent file...")
            with open(filepath,"r") as f:
                objectives=[TaskObjective.parse_raw(line) for line in f.readlines()]
        
        return adapter.to_rl_dataset(objectives,tokenizer=tokenizer,config=config,processor=processor)
    

    def _step_explore_batch(self, tasks: Sequence[Task]):
        with ThreadPoolExecutor(max_workers=self._num_exploration_threads) as executor:
            # TODO: I have no idea what data_id and rollout_id are.
            futures = [
                executor.submit(self._step_explore, task, "unknown data_id", "unknown rollout_id")
                for task in tasks
            ]
            results = [future.result() for future in futures]
            return results

    def _step_explore(self, task: Task, data_id: str, rollout_id: str):
        """
        Step 1: explore the environment to find out possible actions and their results.
        """
        # reset env every time
        env_worker = Explorer(
            env_type=task.env_type,
            task_id=task.task_id,
            instance_id=None,
            env_service_url=self._env_service_url,
        )
        llm_chat_fn = self._get_llm_chat_fn(
            sampling_params={
                "temperature": self._exploration_llm_temperature,
                "top_p": self._exploration_llm_top_p,
                "top_k": self._exploration_llm_top_k,
            }
        )
        agent_flow: BaseAgentFlow = AgentFlow(
            enable_context_generator=False,
            llm_chat_fn=llm_chat_fn,
            tokenizer=self._tokenizer,
            config=self._config,
        )
        agent_flow.max_steps = self._max_explore_step  # TODO(cc): this is ugly

        old_objectives = self._old_retrival.retrieve_objectives(task)

        traj = env_worker.execute(
            data_id=data_id,
            rollout_id=rollout_id,
            system_prompt=get_agent_interaction_system_prompt(task, old_objectives),
            agent_flow=agent_flow,
        )

        return traj

    def _step_summarize_batch(
        self, tasks: Sequence[Task], trajectories: Sequence[Trajectory]
    ) -> list[TaskObjective]:
        with ThreadPoolExecutor(max_workers=self._num_exploration_threads) as executor:
            futures = [
                executor.submit(self._step_summarize, task, traj)
                for task, traj in zip(tasks, trajectories)
            ]
            results = [future.result() for future in futures]
            results = sum(results, [])

            # append to old retrival, to avoid duplicate exploration next time.
            for r in results:
                self._old_retrival.add_objective(r)

            return results

    def _step_summarize(
        self, task: Task, trajectory: Trajectory
    ) -> list[TaskObjective]:
        """
        Step 2: summarize the results of the exploration to generate the TASK (query and gt).

        Args:
            task: Task
            trajectories: Trajectory.
        """
        # 这个方法从现在看基本上是固定的
        llm_fn = self._get_llm_chat_fn()
        old_objectives = self._old_retrival.retrieve_objectives(task)
        system_prompt, user_prompt = get_task_summarize_prompt(
            [trajectory], old_objectives, len_history=self._task_summary_history_length
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        llm_output = llm_fn(messages=messages)["content"]
        tasks = parse_tasks_from_response(task, llm_output)
        return tasks

    def _get_llm_chat_fn(self, sampling_params: Optional[dict] = None) -> Callable:
        def llm_chat(
            messages: list[dict[str, str]],
            custom_sampling_params: Optional[dict] = None,
            request_id: Optional[str] = None,
        ) -> dict:
            """
            input messages: [{"role": "system", "value": "..."}, {"role": "user", "value": "..."}]
            output messages: [{"role": "assistant", "value": "..."}]
            """
            updated_sampling_params = {}
            if sampling_params:
                updated_sampling_params.update(sampling_params)
            if custom_sampling_params:
                updated_sampling_params.update(custom_sampling_params)

            # output_messages = []
            input_messages = copy.deepcopy(messages)
            res = None
            for i in range(self._max_llm_retries):
                try:
                    res = self._llm_client.chat(
                        messages=input_messages, sampling_params=updated_sampling_params
                    )
                    break

                except Exception as e:
                    logger.exception(f"rollout_server.{i} error: {e.args}")
                    time.sleep(i + 1)

            assert res is not None, f"LLM client failed to chat"
            return {
                "role": "assistant",
                "content": res,
            }

        return llm_chat


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
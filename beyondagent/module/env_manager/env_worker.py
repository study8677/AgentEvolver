import uuid

from omegaconf import DictConfig

from beyondagent.client.env_client import EnvClient
from beyondagent.module.agent_flow.base_agent_flow import BaseAgentFlow
from beyondagent.schema.task import Task
from beyondagent.schema.trajectory import Trajectory
from loguru import logger

class EnvWorker(object):

    def __init__(self, task: Task, instance_id: str = None, thread_index: int = None,
                 config: DictConfig = None):
        self.env = EnvClient(base_url=config.env_service.env_url)
        self.task = task
        self.env_type: str = task.env_type
        self.task_id: str = task.task_id
        self.instance_id: str = instance_id if instance_id is not None else uuid.uuid4().hex
        self.thread_index: int = thread_index

    def execute(self, data_id: str, rollout_id: str, add_exp: bool, task_train_exp_mode: str,agent_flow: BaseAgentFlow, **kwargs) -> Trajectory:    # add add_exp & task_train_exp_mode by ANNI
        trajectory: Trajectory = Trajectory(data_id=data_id, rollout_id=rollout_id, steps=[], query="")

        try:
            init_response = self.env.create_instance(env_type=self.env_type,
                                                    task_id=self.task_id,
                                                    instance_id=self.instance_id)
        except Exception as e:
            logger.exception(f"encounter exception in env_worker.create_instance~ error={e.args}")
            return trajectory
        
        

        try:
            state_message: list[dict] = init_response["state"]
            assert isinstance(state_message,list) and len(state_message)==2, "state_message must be list and its length must be 2"
            # replace query if new query is in task
            if self.task.query is not None:
                assert state_message[-1]["role"] == "user", "the latest message from environment must be user query"
                state_message[-1]["content"] = self.task.query
            trajectory: Trajectory = Trajectory(data_id=data_id,
                                                rollout_id=rollout_id,
                                                steps=state_message,
                                                query=state_message[-1]["content"])
            trajectory: Trajectory = agent_flow.execute(trajectory=trajectory, env=self.env, instance_id=self.instance_id, add_exp=add_exp, task_train_exp_mode=task_train_exp_mode, # add add_exp & task_train_exp_mode by ANNI
                                                        **kwargs)
        except Exception as e:
            logger.exception(f"encounter exception in env_worker.agent_flow~ error={e.args}")

        try:
            self.env.release_instance(self.instance_id)
        except Exception as e:
            logger.exception(f"encounter exception in env_worker.release_instance~ error={e.args}")

        return trajectory

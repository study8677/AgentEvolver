import json
import time
from typing import Optional, cast

from loguru import logger

from beyondagent.client.em_client import EMClient
from beyondagent.client.env_client import EnvClient
from beyondagent.module.agent_flow.base_agent_flow import BaseAgentFlow
from beyondagent.module.agent_flow.reward_calculator import RewardCalculator
from beyondagent.schema.trajectory import Trajectory
from beyondagent.utils.utils import convert_tool_to_user_message, clip_state_content_correctly

from beyondagent.module.task_manager.rewards import LlmAsJudgeRewardCalculator,LlmAsJudgeRewardCalculatorWithGT,EnvGrader

class AgentFlow(BaseAgentFlow):

    def __init__(self,reward_calculator:Optional[RewardCalculator]=None, **kwargs):
        super().__init__(**kwargs)
        self._reward_calculator = reward_calculator
        if self._reward_calculator is not None:
            logger.info(f"reward_calculator={self._reward_calculator}")
        else:
            logger.info(f"reward_calculator=env")
        self._enable_context_generator=self.config.experience_maker.enable_context_generator

        self.instruction_template_ids = self.tokenizer.encode("<|im_start|>user\n")
        self.response_template_ids = self.tokenizer.encode("<|im_start|>assistant\n")
        self.em_client = EMClient(base_url=self.config.experience_maker.base_url)

    def execute(self, trajectory: Trajectory, env: EnvClient, instance_id: str, **kwargs) -> Trajectory:
        # TODO refactor this
        if isinstance(self._reward_calculator,EnvGrader):
            self._reward_calculator.set_instance_id(instance_id)
        
        # In some cases, context_generator will be disabled by setting self._enable_context_generator to False.
        if self._enable_context_generator:
            history_experience = self.em_client.call_context_generator(
                trajectory=trajectory,
                retrieve_top_k=self.config.experience_maker.retrieve_top_k,
                workspace_id=self.config.experience_maker.workspace_id)

            if history_experience:
                logger.info(f"history_experience={history_experience}")
                new_content = history_experience + "\n\n" + trajectory.steps[-1]["content"]
                trajectory.steps[-1]["content"] = new_content
            else:
                logger.info(f"history_experience is empty!")

        request_id: str = ""
        for act_step in range(self.max_steps):
            # if use qwen3, add /no_think
            if self.config.actor_rollout_ref.rollout.use_qwen3:
                trajectory.steps[-1]["content"] += " /no_think"

            prompt_text = self.tokenizer.apply_chat_template(trajectory.steps, 
                                                             tokenize=False,
                                                             add_generation_prompt=True)
            current_token_len = len(self.tokenizer(prompt_text, return_tensors="pt", padding=False)["input_ids"][0])

            # yunpeng 0623: to prevent add an imend token to an uncompleted seq, 
            # because the message-type output will be applied chat_template.
            max_response_length = self.config.actor_rollout_ref.rollout.response_length
            if current_token_len + max_response_length > self.max_model_len:
                logger.warning(f"exceed max model len={self.max_model_len}")
                break

            enable_request_id = False
            if hasattr(self.config.actor_rollout_ref.rollout, "enable_request_id"):
                enable_request_id: bool = self.config.actor_rollout_ref.rollout.enable_request_id
                assert isinstance(enable_request_id, bool), f"enable_request_id is bool value"

            t_start = time.time()
            request_id = request_id if enable_request_id else None
            # callback llm server, messages.size=1
            llm_output: dict = {}
            try:
                llm_output = self.llm_chat_fn(trajectory.steps, request_id=request_id)
            except Exception as e:
                logger.exception(f"call llm_chat_fn error with {e}")
                break

            time_cost = round(time.time() - t_start, 4)
            new_request_id: str = llm_output.pop("request_id", "")

            info_dict = {
                "act_step": act_step,
                "llm_output": llm_output,
                "new_request_id": new_request_id,
                "request_id": request_id,
                "time_cost": time_cost,
            }
            # logger.info(f"info_dict={json.dumps(info_dict)}")

            request_id = new_request_id
            trajectory.steps.append(llm_output)

            try:
                env_output = env.step(instance_id, llm_output)
                env_messages: list[dict] = env_output["state"]
            except Exception as e:
                logger.exception(f"call env.step error with {e}")
                break
            # convert role_tool to role_user message
            # breakpoint()
            
            # useless: for tool role
            assert len(env_messages)>0, "env returns empty messages"
            for env_message in env_messages:
                if env_message["role"] == "tool":
                    env_message = cast(dict, convert_tool_to_user_message(env_message, format="qwen"))
                
                state_content: str = env_message["content"]
                
                env_message["content"] = clip_state_content_correctly(
                    self.tokenizer, 
                    state_content,
                    self.max_env_len
                )
                

                trajectory.steps.append(sanitize_env_state(env_message))
            trajectory.is_terminated = env_output["is_terminated"]
            # TODO require env
            # trajectory.reward.outcome = env_output["reward"]["outcome"]
            # trajectory.reward.description = env_output["reward"]["description"]
            # trajectory.reward.outcome = env_output["reward"]
            # trajectory.reward.description = "Outcome 1 = success, 0 = failure."

            if trajectory.is_terminated:
                break
        if self._reward_calculator is not None:
            score = self._reward_calculator.calculate_reward(trajectory, env)
        else:
            score = env.evaluate(instance_id, params={"sparse": True})
        trajectory.reward.outcome = score
        trajectory.reward.description = "Outcome 1 = success, 0 = failure."

        if trajectory.steps[-1]["role"] == "user":
            trajectory.steps = trajectory.steps[:-1]

        return trajectory


def sanitize_env_state(state: dict):
    """
    sanitize env state
    """
    # remove empty tool_calls
    if "tool_calls" in state and not state["tool_calls"]:
        state.pop("tool_calls")
    
    return state
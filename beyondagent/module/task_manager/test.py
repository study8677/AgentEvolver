

import hydra

from beyondagent.client.llm_client import DashScopeClient
from beyondagent.module.task_manager import NaiveTaskObjectiveRetrieval, TaskManager
from beyondagent.schema.task import Task


@hydra.main(
    config_path="../../../config",
    config_name="beyond_agent_dataflow",
    version_base=None,
)
def test(config):
    import transformers
    import json
    from torch.utils.data import DataLoader
    from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True
    )
    manager = TaskManager(
        config,
        DashScopeClient(),
        NaiveTaskObjectiveRetrieval(),
        tokenizer=tokenizer,
        env_service_url="http://localhost:8000",
        max_explore_step=5,
        max_llm_retries=3,
        num_explore_threads=2,
        n=1,
    )
    task = Task(task_id="0a9d82a_1", env_type="appworld")
    tasks = [task] * 100
    dataset = manager.get_dataset(iter(tasks), bs=2, tokenizer=tokenizer, config=config)
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=default_collate_fn
    )

    print("ready to retrieve data")
    for data in dataloader:
        import pdb

        pdb.set_trace()


if __name__ == "__main__":
    test()
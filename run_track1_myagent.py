# run_track1_myagent.py
import os
import json
import logging

from websocietysimulator import Simulator
from websocietysimulator.agent.my_agent import MySimulationAgent
from websocietysimulator.llm import OpenAILLM 

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    task_set = "yelp"  # 或 "goodreads"/"yelp"

    # 1. 从环境变量中读取 API Key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("环境变量 OPENAI_API_KEY 未设置，请在 shell 中先设置再运行。")

    # 2. 初始化 Simulator
    simulator = Simulator(
        data_dir="processed_data",  # 你刚才放 3 个 json 的目录
        device="cpu",              # 先用 cpu
        cache=True
    )

    simulator.set_task_and_groundtruth(
        task_dir=f"./example/track1/{task_set}/tasks",
        groundtruth_dir=f"./example/track1/{task_set}/groundtruth"
    )

    # 3. 设置 Agent & LLM
    simulator.set_agent(MySimulationAgent)
    llm = OpenAILLM(
        api_key=api_key,
        # 如果这个类支持 model_name 之类的参数，可以在这里指定，比如：
        # model_name="gpt-4.1-mini"
    )
    simulator.set_llm(llm)

    # 4. 先跑 20 个任务 sanity check
    outputs = simulator.run_simulation(
        number_of_tasks=200,
        enable_threading=True,
        max_workers=4
    )
    print(f"Ran {len(outputs)} tasks. Example output[0]:")
    print(outputs[0])

    # 5. 评估
    evaluation_results = simulator.evaluate()
    print("=== Evaluation ===")
    print(json.dumps(evaluation_results, indent=2))

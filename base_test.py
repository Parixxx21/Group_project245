from websocietysimulator import Simulator
from example.trackOneSubmission_example.userBehaviorSimulation import MySimulationAgent
from websocietysimulator.llm import OpenAILLM

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPEN_API_KEY")

llm_client = OpenAILLM(
    api_key=api_key,
)

simulator = Simulator(
    data_dir="./example/processed_data",
    device="cpu",
    cache=False
)

simulator.set_task_and_groundtruth(
    task_dir="./example/track1/amazon/tasks",
    groundtruth_dir="./example/track1/amazon/groundtruth"
)

simulator.set_agent(MySimulationAgent)
simulator.set_llm(llm_client)

outputs = simulator.run_simulation(
    number_of_tasks=50, # Should set to None afterwards
    enable_threading=True,
    max_workers=3
)

print("\n==================== RAW OUTPUTS (first 10) ====================")
for i, out in enumerate(outputs[:10]):
    print(f"\n--- Task {i} ---")
    print(out)

results = simulator.evaluate()
print("\nRESULTS:", results)
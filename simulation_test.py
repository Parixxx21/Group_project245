#Test for simulation 
from websocietysimulator import Simulator
from websocietysimulator.agent.my_agent import MySimulationAgent
from websocietysimulator.llm import OpenAILLM
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPEN_API_KEY")

llm_client = OpenAILLM(
    api_key=api_key,
    model="gpt-4.1-mini"
)

simulator = Simulator(
    data_dir="./example/processed_data",
    # Modify this if you are using an environment support GPU
    device="cpu",
    cache=False
)

simulator.set_task_and_groundtruth(
    task_dir="./example/track1/amazon/tasks",
    groundtruth_dir="./example/track1/amazon/groundtruth"
)

simulator.set_agent(MySimulationAgent)
simulator.set_llm(llm_client)

# Modify this if you are using an environment support multi-threading 
outputs = simulator.run_simulation(number_of_tasks=None, enable_threading=False, 
    max_workers=1)
print("\n==================== RAW OUTPUTS ====================")
for i, out in enumerate(outputs[:10]): 
    print(f"\n--- Task {i} ---")
    print(out)

results = simulator.evaluate()

print("RESULTS:", results)

from websocietysimulator import Simulator
from example.trackOneSubmission_example.userBehaviorSimulation import MySimulationAgent
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
    device="auto",
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

results = simulator.evaluate()

print("RESULTS:", results)

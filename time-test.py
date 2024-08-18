import datetime
import subprocess
from pynvml import *
import json
import time
from ruamel.yaml import YAML
from pathlib import Path
from core.waifuHub import WaifuHub
file_name = "mixtral-instruct-q2.yaml"
test_result_folder = Path("test-results")
gpu_layer_steps = 2
gpu_layer_min = 10
gpu_layer_max = 50
model_name = "-".join(file_name.split(".")[:-1])

cur_vram_used = 0.0
prev_vram_used = 0.0
# If the vram doesn't increase, it likely means that you have reach the vram limit.
auto_terminate = True 

i = 1
while True:
    log_name = test_result_folder / (model_name + "-test-" + str(i) + ".json")
    if not log_name.exists():
        break
    i += 1
log_name.parent.mkdir(parents=True, exist_ok=True)
with open(log_name, 'w') as f:
    json.dump([], f)

def start_server():
    subprocess.run(
        ["docker", "compose", "-f" "./servers/LocalAI/docker-compose.yaml", "up", "-d"]
    )
    time.sleep(0.1)

def restart_server():
    subprocess.run(["docker", "compose", "-f" "./servers/LocalAI/docker-compose.yaml", "restart"])
    time.sleep(0.15)

def end_server():
    subprocess.run(["docker", "compose", "-f" "./servers/LocalAI/docker-compose.yaml", "down"])
    time.sleep(0.1)

def change_gpu_layer(gpu_layers: int) -> None:
    model_folder = Path("servers/LocalAI/models")
    model_path = model_folder / file_name

    if not model_path.exists():
        raise FileNotFoundError

    yaml = YAML()
    with model_path.open() as file:
        yaml_file = yaml.load(file)

    yaml_file["gpu_layers"] = gpu_layers

    yaml.dump(yaml_file, model_path)

def get_vram():

    nvmlInit()
    # print(f"Driver Version: {nvmlSystemGetDriverVersion()}")

    deviceCount = nvmlDeviceGetCount()
    gpus_list = []
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(
            i
        )  # This outputs a struct in C. So, don't bother
        name = nvmlDeviceGetName(handle)
        # print(f"Device {i} : {name}")
        info = nvmlDeviceGetMemoryInfo(handle)
        total_memory_gb = info.total / (1024**3)
        # print(f"Total memory: {total_memory_gb} GiB")
        free_memory_gb = info.free / (1024**3)
        # print(f"Free memory: {free_memory_gb} GiB")
        used_memory_gb = info.used / (1024**3)
        # print(f"Used memory: {used_memory_gb} GiB")
        gpus_list.append(
            {
                "device_id": i,
                "device_name": name,
                "used_vram": round(used_memory_gb, 2),
            }
        )

    # total_vram = round(sum([gpu["used_vram"] for gpu in gpus_list]), 2)
    # gpu_equation_str = [ str(round(gpu["used_vram"], 2)) for gpu in gpus_list]
    # output = f"Total vram: {gpu_equation_str} = {total_vram} GiB"
    time.sleep(0.01)
    return gpus_list

def model_setup():
    file_name = "./function_call/function_call_list.json"
    # API_BASE="http://localhost:8080",
    llm_model = WaifuHub(model_name, function_list_file=file_name, username="Max King", llm_provider="local")
    llm_model.load_lore("ayumi_info.json")
    return llm_model

def prompt(llm_model):
    prompt = "Hello! How are you?" # "Do you know what time is it?"
    response = llm_model.chat(prompt)
    return response

def time_prompt(llm_model):
    now = datetime.datetime.now()
    output = prompt(llm_model)
    then = datetime.datetime.now()
    total_time = (then - now).total_seconds()
    return total_time, output

def log_test(json_content):
    with open(log_name, 'r') as f:
        loaded_json = json.load(f)

    with open(log_name, "w") as f:
        loaded_json.append(json_content)
        json.dump(loaded_json,f, indent=4)

def test_process(llm_model, n_gpu_layers):
    data = {}
    total_time, response = time_prompt(llm_model)
    data["model_name"] = model_name
    data["response_1"] = str(response.choices[0].message.content)
    data["init_total_time_1"] = total_time
    total_time, response = time_prompt(llm_model)
    data["response_2"] = str(response.choices[0].message.content)
    data["total_time_2"] = total_time
    data["gpu_layers"] = n_gpu_layers
    gpus_list = get_vram()
    data["gpu_list"] = gpus_list
    data["total_vram"] = round(sum([gpu["used_vram"] for gpu in gpus_list]), 2)
    global prev_vram_used, cur_vram_used
    prev_vram_used = cur_vram_used
    cur_vram_used = data["total_vram"]
    return data

def main():
    i = gpu_layer_min
    change_gpu_layer(i)
    end_server()
    start_server()
    llm_model = model_setup()
    print("Start Testing")
    while i < gpu_layer_max:
        print(f"GPU layers: {i}")
        change_gpu_layer(i)
        start_server()
        result = test_process(llm_model, i)
        log_test(result)
        i += gpu_layer_steps
        if auto_terminate and cur_vram_used == prev_vram_used and prev_vram_used != 0.0 and cur_vram_used != 0.0:
            return
        end_server()
        
    if i >= gpu_layer_max:
        i = gpu_layer_max
        change_gpu_layer(i)
        start_server()
        print(f"GPU layers: {i}")
        result = test_process(llm_model, i)
        log_test(result)


try:
    main()
except Exception as e:
    print(f"There was an error: {e}")
finally:
    end_server()
# Hosting Airflow with Local LLMs

## Goal of This Project
This project is intended to serve as a starter project for working with locally hosted Large Language Models (LLMs) using LLama.Cpp. It focuses on running Airflow DAGs that can utilize these locally hosted LLMs in conjunction with Langchain.

## Components Provided by This Project
1. Airflow integrated with Celery and Redis.
2. PostgreSQL 15.
3. Python 3.11 including libraries such as PyTorch, Langchain, OpenAI, pandas, etc.
4. Dockerized Python [llama.cpp server](https://github.com/abetlen/llama-cpp-python) with support for Nvidia GPU, accessible from Airflow.
5. Scripts for building and running Docker containers.
6. DAG example for connecting with a locally hosted LLM from Airflow using Langchain.

## Prerequisites
1. Nvidia GPU(s) with sufficient VRAM to run the desired models.
2. Nvidia drivers with CUDA support for your system.
3. Run `nvidia-smi` to verify that the drivers are working correctly.


```shell
(base) ➜  ~ nvidia-smi
Tue Dec 26 14:02:31 2023       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03             Driver Version: 535.129.03   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3090        On  | 00000000:01:00.0  On |                  N/A |
|  0%   29C    P8              38W / 350W |    513MiB / 24576MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce RTX 3090        On  | 00000000:02:00.0 Off |                  N/A |
|  0%   25C    P8              17W / 370W |     12MiB / 24576MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      3494      G   /usr/lib/xorg/Xorg                          152MiB |
|    0   N/A  N/A      3634      G   /usr/bin/gnome-shell                        144MiB |
|    0   N/A  N/A      5000      G   ...irefox/3600/usr/lib/firefox/firefox      141MiB |
|    0   N/A  N/A      6614      G   ...sion,SpareRendererForSitePerProcess       57MiB |
|    1   N/A  N/A      3494      G   /usr/lib/xorg/Xorg                            4MiB |
+---------------------------------------------------------------------------------------+

```

4. `Python 3.11` installed.
5. Installed `docker` with configured sudo-less docker user. For Docker installation on Ubuntu 22.04, refer to this [DigitalOcean tutorial](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-22-04).
6. Installed `docker-compose`. For Docker Compose installation on Ubuntu 22.04, see this [DigitalOcean tutorial](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-compose-on-ubuntu-22-04).
7. Installed `poetry`, used for local type checking and DAG development.
8. Model in `GUFF` format. A good place to find models is [TheBloke on Hugging Face](https://huggingface.co/TheBloke).

__Warning__: _You may need to adapt the Nvidia image version used in `LocalLLamaCPPServerDockerfile` to match the CUDA version compatible with your driver. In my example, it is `CUDA Version: 12.2`_.

## How to Start
### Run Airflow with LLama.cpp
1. Download this repository or create a new one using this template.
2. Download the chosen LLM model in `GUFF` format (or any other format supported by LLama.Cpp) and place it in the `models` folder.
3. Modify `LocalLLamaCPPServerDockerfile` to include the correct model name and the number of layers to be passed to the GPU in the `ENTRYPOINT` section.
4. To add extra Python packages, use `poetry add [name of package]` or modify `pyproject.toml` and then run `poetry install` and `poetry update`. Select the created poetry environment in your IDE for type checking.
5. Run `./build_all.sh` after making it executable (use `chmod +x name_of_script` for all scripts, e.g., `sudo chmod +x build_all.sh`).
6. After the build is complete, there should be a Docker image `llm-server` containing the dockerized `llama.cpp` server with GPU support, and an `extending_airflow` image, containing Airflow extended with chosen Python libraries.
7. To run everything, execute `./start_all.sh` and to stop it, use `./stop_all.sh`.
8. Open a browser and navigate to http://0.0.0.0:8080 to launch the Airflow webserver. Log in with username: `airflow` and password: `airflow`.
9. Refer to `dags/test_connection_to_local_llm.py` and `dags/test_multi_connection_to_local_llm_4x.py` as starting points.

### How to modify number of llama.cpp servers
The script `run_llm_servers_for_data_generation.sh` provides parameters to configure how many llama.cpp servers should be deployed

The default values:
```bash
num=4 # number of servers to deploy use -n flag to change
model="models/llama-2-13b-chat.Q5_K_M.gguf" # llm model to deploy with llama.cpp server, use -m flat
split=0 # if 0 models will load between 2 gpus without spliting, if 1 models will be splited between all gpus, use -s flag
n_threads=1 # number of threads for every llama.cpp server, use -t flag

```

Example how to run:
```bash
./run_llm_servers_for_data_generation.sh -n 4 -m models/llama-2-13b-chat.Q4_K_M.gguf -t 8 -s 0
```

You can modify it in `./start_all.sh` script

You can also manually stop just inference servers by running command `./stop_llm_servers_for_data_generation.sh`


### Using Just the Dockerized Model Without Airflow
1. Download the chosen LLM model in `GUFF` format (or any other format supported by LLama.Cpp) and place it in the `models` folder.
2. Modify `LocalLLamaCPPServerDockerfile` to include the correct model name and the number of layers to be passed to the GPU in the `ENTRYPOINT` section.
3. To add extra Python packages, use `poetry add [name of package]` or modify `pyproject.toml` and then run `poetry install` and `poetry update`. Select the created poetry environment in your IDE for type checking.
4. Run `./build_llm_server.sh` to build the dockerized version of the LLama.cpp server with GPU support.
5. Execute `./run_llm_server.sh` and `docker kill llm-server` and to stop it. Server will run on `5556` port, you can check loaded models `http://localhost:5556/v1/models` or documentation `http://localhost:5556/docs#/`
6. See `run_completion_on_local_llama.py` as a starting point for development without Airflow. You can run it within the Poetry environment.


### Create `.env` for Airflow
Example of an `.env` file:
```
AIRFLOW_UID=1000
AIRFLOW_GID=0
```

## Workflow
1. After installation, connect the poetry environment to your IDE for type checking.
2. Consider using `DBeaver` or a similar tool to create an extra database in the provided `PostgreSQL` and utilize `airflow postgres hooks` for communication from within `Airflow DAGs`.
3. Every `DAG` created in the `dags` folder will be visible and usable in `Airflow`.
4. The `nvtop` package can be used to monitor GPU usage. See [nvtop on GitHub](https://github.com/Syllo/nvtop).
5. Volumes are already created for storing SQL scripts and raw data (`sql` and `data`). For more complex projects, consider integrating with a service like `S3`.

## Known Limitations
1. Currently, this template does not support Kubernetes, but it could be added relatively easily.
2. The template uses default passwords for Airflow and PostgreSQL, which should be changed in `docker-compose.yaml`.

## Credits
Parts of this template were created based on this tutorial: [coder2j's YouTube tutorial](https://www.youtube.com/watch?v=K9AnJ9_ZAXE) and [coder2j's GitHub repo](https://github.com/coder2j/airflow-docker), along with insights from this [Reddit thread](https://www.reddit.com/r/LocalLLaMA/comments/17ffbg9/using_langchain_with_llamacpp/).


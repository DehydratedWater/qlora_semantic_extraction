# Synthetic Data Creation Tool for Model Refinement

## Goal of This Project
This repository is part of a larger project. The core idea of the project aims at building a system that extracts entities from text with semantic relations between them, and uses these entities for question answering with retrieval. By maintaining a graph of all semantic relations, it is possible to improve the results of the retrieval and make it not only precise but also deep, by providing a way to reason about second, third, ... order causes and effects.

### This Repository
This part of the project is intended for generating synthetic data that may be used for fine-tuning LLM models. The main use for this data is to prepare smaller, fine-tuned models, for increasing speed and lowering the cost of extracting relations and entities from text.

### Hugging Face
The dataset generated with this repository, as well as a backup of the full PostgreSQL database, is available at [DehydratedWater42/semantic_relations_extraction](https://huggingface.co/datasets/DehydratedWater42/semantic_relations_extraction)

## Generated Data
### Generation Process
This data was generated based on the `datasets/scientific_papers` dataset. This dataset contains a list of scientific articles with separate `abstracts` and `lists of contents`. Here is the synthetic data generation overview:

1. All the `abstracts` and `lists of contents` were inserted into the database.
2. The main content of every article was split into overlapping segments of 1k LLaMA tokens with a 200-token overlap.
3. 10k of the `abstracts` + `lists of contents` were summarized by LLaMA 13b.
4. Generated `summaries` + `split text segments` were transformed by LLaMA 13b into unprocessed JSONs.
5. All generated JSONs were validated and cleaned up.
6. Validated JSONs were reformatted into datasets that may be used for fine-tuning.

### Example of output data
```json
{
    "section_description": "The article discusses the current reversal phenomenon in a classical deterministic ratchet system. The authors investigate the relationship between current and bifurcation diagrams, focusing on the dynamics of an ensemble of particles. They challenge Mateos' claim that current reversals occur only with bifurcations and present evidence for current reversals without bifurcations. Additionally, they show that bifurcations can occur without current reversals. The study highlights the importance of considering the characteristics of the ensemble in understanding the behavior of the system. The authors provide numerical evidence to support their claims and suggest that correlating abrupt changes in the current with bifurcations is more appropriate than focusing solely on current reversals.",
    "list_of_entities": [
        "reversals",
        "mateos",
        "figures",
        "rules",
        "current_reversal",
        "ensemble",
        "bifurcation",
        "jumps",
        "thumb",
        "spikes",
        "current",
        "particles",
        "open_question",
        "behavior",
        "heuristics",
        "direction",
        "chaotic",
        "parameter"
    ],
    "relations": [
        {
            "description": "bifurcations in single - trajectory behavior often corresponds to sudden spikes or jumps in the current for an ensemble in the same system",
            "source_entities": [
                "bifurcation"
            ],
            "target_entities": [
                "current"
            ]
        },
        {
            "description": "current reversals are a special case of this",
            "source_entities": [
                "current"
            ],
            "target_entities": [
                "bifurcation"
            ]
        },
        {
            "description": "not all spikes or jumps correspond to a bifurcation",
            "source_entities": [
                "spikes"
            ],
            "target_entities": [
                "bifurcation"
            ]
        },
        {
            "description": "the open question is clearly to figure out if the reason for when these rules are violated or are valid can be made more concrete",
            "source_entities": [
                "current"
            ],
            "target_entities": [
                "open_question"
            ]
        }
    ]
}
```

### Expected output JSON schema
```json
{
  "$schema": "extraction_schema.json",
  "type": "object",
  "properties": {
    "section_description": {
      "type": "string"
    }
    "list_of_entities": {
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "relations": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "description": {
            "type": "string"
          },
          "source_entities": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "target_entities": {
            "type": "array",
            "items": {
              "type": "string"
            }
          },
          "strength": {
            "type": "string",
            "enum": ["strong", "moderate", "weak"]
          }
        },
        "required": ["description", "source_entities", "target_entities"]
      }
    },
    
  },
  "required": ["list_of_entities", "relations", "section_description"]
}
```

## Decisions
1. I used Airflow because I had previously prepared a template that allowed me to run Airflow pipelines with local LLMs.
2. There is a whole section of the database with extracted relations and entities, mostly for estimating the connectivity and scale of the extracted data.
3. The final dataset is being created with `jupyters/02_export_data.ipynb`; it was just quicker than adding a new volume to the docker-compose.
4. I chose `datasets/scientific_papers` as it already provided a good base for summaries (i.e., Abstracts) and did not require me to iteratively summarize all the contents, which would require additional time.
5. This project does not use ChatGPT or other external APIs; all processing was done locally on 2x3090RTX + some OrangePIs. The goal is to generate a fine-tuned model that can be hosted more cheaply, and also provide the same utility as this two-step LLaMA 13b process. OpenAI does not allow using the results of generation for fine-tuning other models; hence, all this data was generated locally with LLaMA 2, as the license permits improving LLaMA 2 with data generated with LLaMA 2. This is not perfect, but as long as I use `datasets/scientific_papers`, there is still the issue of licensing; it all will need to be regenerated in the future with a more open stack.
6. The goal is to create a small 3B-7B model that can be used for the task of extracting entities and semantic relations, which may be run on a small ARM board like OrangePI, with minimal cost at a reasonable speed.
7. I used LLaMA 2 Chat because, in the past, I was able to achieve the most stable results with that model.
8. I set the temperature to 0.7 to allow the model to infer some missing information and generate better summaries, but the trade-off of using a non-zero temperature is more involved result cleanup. Still, almost 88% of the generated data had a fixable structure.

## Future Plans for the Project
1. Fine-tune LLaMA 2 7B with synthetic data (try and evaluate the speed and quality of generation).
2. Generate more synthetic data, clean it, and fine-tune the model further.
3. Build a system for mixed querying of the data (I've built a prototype; now, I would like to recreate it as a whole standalone service).
4. After running it successfully, regenerate data based on the Wikipedia dataset or another fully open-source dataset, and replace LLaMA with a truly open-source model.


## Statistics
1. I ran the generation on 4 instances of LLaMA 2-chat on 2x3090RTX + i7 4790K. The processing averaged around 1 result per minute (either a summary or JSON). The whole process, excluding coding and experimentation, took approximately 20,000 minutes, which is roughly 14 days of compute time, and required about 120 kWh of power. In the near future, I need to upgrade the CPU + RAM to remove that bottleneck.
```bash
./run_llm_servers_for_data_generation.sh -n 4 -t 1 -m "models/llama-2-13b-chat.Q4_K_M.gguf" -c 4096 -b 1512
```
2. I tested hosting on ARM boards; a 13b model quantized to q4 was able to be hosted with stable speed for an extended time, achieving a speed of 2.34 tokens/s per one OrangePI. With an RTX 3090 paired with my somewhat outdated CPU, an i7 4790K, I was able to achieve up to 20 tokens/s. I have 5 OrangePIs 5 16GB, and by running models on all of them, I achieved around 11.7 tokens/s for approximately 50W of power.


## Running the project
### Components Provided by This Project
1. Airflow integrated with Celery and Redis.
2. PostgreSQL 15.
3. Python 3.11 including libraries such as PyTorch, Langchain, OpenAI, pandas, etc.
4. Dockerized Python [llama.cpp server](https://github.com/abetlen/llama-cpp-python) with support for Nvidia GPU, accessible from Airflow.
5. Scripts for building and running Docker containers.
6. DAG example for connecting with a locally hosted LLM from Airflow using Langchain.

### Prerequisites
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

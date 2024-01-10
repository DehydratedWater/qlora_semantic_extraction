#!/usr/bin/bash


num=4
model="models/llama-2-13b-chat.Q5_K_M.gguf"
split=0
n_threads=1

parent_dir_name=$PWD
last=$(echo $parent_dir_name | awk -F'/' '{print $NF}')
network=$last"_connection_to_airflow"

# models/llama-2-13b-chat.Q5_K_M.gguf -> n_threads=1
# models/llama-2-13b-chat.Q4_K_M.gguf -> n_threads=8-12

while getopts n:m:s:t: flag
do
    case "${flag}" in
        n) num=${OPTARG};;
        m) model=${OPTARG};;
        s) split=${OPTARG};;
        t) n_threads=${OPTARG};;
    esac
done

echo "Parameter num: $num"
echo "Parameter model: $image"
echo "Parameter split: $split"
echo "Parameter n_threads: $n_threads"
# docker run -it --entrypoint "python3 your_image_name -m llama_cpp.server --model models/llama-2-13b-chat.Q4_K_M.gguf --n_gpu_layers 200 --port 5556 --host 0.0.0.0"
# entrypoint="python3 -m llama_cpp.server --model $image --n_gpu_layers 200 --port 5556 --host 0.0.0.0"


# ENV N_CTX=${N_CTX:-4098}
# ENV N_BATCH=${N_BATCH:-1024}

for i in $(seq 1 $num)
do
    echo "Starting docker llm-server-$i"
    docker remove "/llm-server-$i"
    # memlock=16384:16384
    if [ "$split" -eq 1 ]; then
        docker run -e MODEL_NAME=$model -e N_THREADS=$n_threads -e N_BATCH=2048 -e N_CTX=4096 --gpus all --ulimit memlock=16384:16384 --network $network -d -v "$(pwd)/models:/app/models" --name "llm-server-$i" llm-server
    else
        mod=$(($i % 2))
        echo "mod=$mod"
        device="device=$mod"
        docker run -e MODEL_NAME=$model -e N_THREADS=$n_threads -e N_BATCH=2048 -e N_CTX=4096 --gpus $device --ulimit memlock=16384:16384 --network $network -d -v "$(pwd)/models:/app/models" --name "llm-server-$i" llm-server
    fi
    
done
# num=${1:-4}
# image=${2:-"models/llama-2-13b-chat.Q4_K_M.gguf"}

# echo "Parameter A: $num"
# echo "Parameter B: $image"



# docker run --gpus all --network qlora_semantic_extraction_connection_to_airflow -d -v "$(pwd)/models:/app/models" --name llm-server-1 llm-server
# docker run --gpus all --network qlora_semantic_extraction_connection_to_airflow -d -v "$(pwd)/models:/app/models" --name llm-server-2 llm-server
# docker run --gpus all --network qlora_semantic_extraction_connection_to_airflow -d -v "$(pwd)/models:/app/models" --name llm-server-3 llm-server
# # docker run --gpus all --network qlora_semantic_extraction_connection_to_airflow -d -v "$(pwd)/models:/app/models" --name llm-server-1 llm-server
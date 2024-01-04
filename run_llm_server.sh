#!/usr/bin/bash

docker remove llm-server

docker run --gpus all --network qlora_semantic_extraction_connection_to_airflow -d -v "$(pwd)/models:/app/models" -p 5556:5556 --name llm-server llm-server
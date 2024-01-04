#!/bin/bash

# This script is used to build all the docker images needed for the project.
# It is meant to be run from the root directory of the project.

./export_requirements.sh

./make_needed_folders.sh

# Build the airflow docker image
cd airflow_docker && ./build_extra_docker.sh && cd ..


# Build the local llama server docker image
cd inference_server_dockers/llama_cpp_server && ./build_llm_server.sh && cd ../..




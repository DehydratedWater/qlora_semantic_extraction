#!/usr/bin/bash

dir=$(dirname "$1")
echo "$dir"
parent_dir_name=$PWD
echo "$parent_dir_name"

last=$(echo $parent_dir_name | awk -F'/' '{print $NF}')
network=$last"_connection_to_airflow"
echo "network=$network"


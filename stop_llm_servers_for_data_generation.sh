#!/bin/bash

name_pattern="llm-server-"

# Corrected variable assignment and command substitution
dockers=$(docker ps -f "name=$name_pattern*" --format "{{.Names}}")

if [[ -n $dockers ]]; then
    echo "dockers is not empty"
    # Iterate over each line in the output
    for i in $dockers
    do
        echo "Killing docker $i"
        docker kill "$i"
        docker remove "$i"
    done
else
    echo "There is no llm-servers docker running"
fi

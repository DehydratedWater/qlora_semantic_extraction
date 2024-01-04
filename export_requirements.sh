#!/bin/bash

poetry export --without-hashes --format=requirements.txt > airflow_docker/requirements.txt
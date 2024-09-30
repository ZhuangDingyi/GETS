#!/bin/bash

calibration_methods=("GETS" "VS" "TS" "ETS" "CaGCN" "GATS")

yaml_directory="./config"
log_directory="./log"

mkdir -p "$log_directory"

yaml_files=("citeseer.yaml" "computers.yaml" "cora-full.yaml" "cora.yaml" "cs.yaml" 
            "ogbn-arxiv.yaml" "photo.yaml" "physics.yaml" "pubmed.yaml" "reddit.yaml")

if ! command -v ~/yq &> /dev/null
then
    echo "Error: yq is not installed. Please install yq (https://mikefarah.gitbook.io/yq/) or choose any other methods that could automatically replace values in yaml files in your machine."
    exit 1
fi

run_script() {
  local yaml_file="$1"
  local method="$2"
  local gpu_id="$3"

  dataset_name=$(basename "$yaml_file" .yaml)

  ~/yq eval ".calibration.calibrator_name = \"$method\"" -i "$yaml_directory/$yaml_file"

  log_file="$log_directory/${dataset_name}_${method}.txt"

  echo "Running: python main.py --dataset=$dataset_name --gpu=$gpu_id --n_runs=10 with $method on GPU $gpu_id"
  CUDA_VISIBLE_DEVICES=$gpu_id python main.py --dataset="$dataset_name" --gpu="$gpu_id" --n_runs=10 > "$log_file" 2>&1
}

process_dataset() {
  local yaml_file="$1"
  local gpu_id="$2"

  for method in "${calibration_methods[@]}"
  do
    run_script "$yaml_file" "$method" "$gpu_id"
  done
}

gpu_id=0
job_count=0

for yaml_file in "${yaml_files[@]}"
do
  process_dataset "$yaml_file" "$gpu_id" &

  gpu_id=$(( (gpu_id + 1) % 10 ))

  job_count=$((job_count + 1))

  if [ "$job_count" -ge 10 ]; then
    wait
    job_count=0
  fi
done

wait
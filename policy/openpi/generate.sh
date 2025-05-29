#!/bin/bash
# Set HuggingFace cache directories
export HF_HOME="/nvmessd/ssd_share/tengbo/cache"
export HUGGINGFACE_HUB_CACHE="/nvmessd/ssd_share/tengbo/cache"
export TRANSFORMERS_CACHE="/nvmessd/ssd_share/tengbo/cache"
export HF_DATASETS_CACHE="/nvmessd/ssd_share/tengbo/cache"

data_dir=${1}
repo_id=${2}
uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py --raw_dir $data_dir --repo_id $repo_id

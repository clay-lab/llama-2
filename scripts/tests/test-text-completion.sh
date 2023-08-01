#!/bin/bash

#SBATCH --job-name=test-llama-2-text
#SBATCH --output=joblogs/%x_%j.txt
#SBATCH --mem-per-cpu=128G
#SBATCH --time=02:00:00


module load miniconda

source activate llama-2

time python \
    example_text_completion.py \
    --ckpt_dir facebook/llama-2/llama-2-7b \
    --tokenizer_path facebook/llama-2/tokenizer.model \
    --max_batch_size 2

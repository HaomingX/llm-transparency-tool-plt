#!/bin/bash
set -e

# path to your ckpt directory
outer_target_dir="./memory/"

if [ ! -d "$outer_target_dir" ]; then
    echo "outer target directory does not exist: $outer_target_dir"
    exit 1
fi
# process llama2
for outer_dir in "$outer_target_dir"/llama2*; do
    if [ -d "$outer_dir" ]; then
        outer_dir_name=$(basename "$outer_dir")
        
        echo "processing outer directory: $outer_dir_name"

        # *-full means merged ckpt
        for full_checkpoint_dir in "$outer_dir"/*-full; do
            if [ -d "$full_checkpoint_dir" ]; then
                checkpoint_name=$(basename "$full_checkpoint_dir")

                echo "processing $checkpoint_name..."

                python run.py --model_path "$full_checkpoint_dir" --model_name "meta-llama/llama-2-7b-hf" --dataset_path "path to your dataset" --save_path "./output/${checkpoint_name}.json" 
            else
                if [ "$full_checkpoint_dir" == "${outer_dir}/*"-full ]; then
                    echo "No '-full' directories found in $outer_dir"
                fi
            fi
        done
    else
        if [ "$outer_dir" == "${outer_target_dir}/llama2*" ]; then
            echo "No 'llama2*' directories found in $outer_target_dir"
        fi
    fi
done
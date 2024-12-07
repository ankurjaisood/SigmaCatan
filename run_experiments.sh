#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e

# Dataset directory:
DATASET_DIR="./datasets/2024-12-04_10_31_09"

# Parameter table:
# Each line defines: reward_func gamma num_epochs target_update_freq loss_func end_turn_penalty tau
params=(
    "BASIC 0.90 1 10000 huber 2.5 0.001"
)

# Output directory for logs
output_dir="./experiment_logs"
mkdir -p "$output_dir"

# Iterate through the parameter table and run `main.py` with the respective arguments
for param in "${params[@]}"; do
    # Read the parameters into variables
    read -r reward_func gamma num_epochs target_update_freq loss_func end_turn_penalty tau <<< "$param"

    # Generate a unique filename for the output
    timestamp=$(date +%Y%m%d_%H%M%S)
    log_file="${output_dir}/${timestamp}_rewardfunc-${reward_func}_gamma-${gamma}_epochs-${num_epochs}_updatefreq-${target_update_freq}_loss-${loss_func}.log"

    # Print the current configuration
    echo "Running with parameters: "
    echo "--static_board --reward_func $reward_func --gamma $gamma --num_epochs $num_epochs --target_update_freq $target_update_freq --loss_func $loss_func --end_turn_penalty $end_turn_penalty --tau $tau"
    echo "Logging output to: $log_file"

    # Run the Python script and redirect output to the log file
    python main.py \
        $DATASET_DIR \
        --static_board \
        --reward_func "$reward_func" \
        --gamma "$gamma" \
        --num_epochs "$num_epochs" \
        --target_update_freq "$target_update_freq" \
        --loss_func "$loss_func" \
        --end_turn_penalty "$end_turn_penalty" \
        --tau "$tau" \
        > "$log_file" 2>&1

    # Check if the Python script exited successfully
    if [ $? -ne 0 ]; then
        echo "Error: main.py failed for parameters: --reward_func $reward_func --gamma $gamma --num_epochs $num_epochs --target_update_freq $target_update_freq --loss_func $loss_func"
        echo "Check the log file: $log_file for details."
        exit 1
    fi

done

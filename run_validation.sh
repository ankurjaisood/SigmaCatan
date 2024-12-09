#!/bin/bash

# Ensure the script exits if any command fails
set -e

# Run the command 4 times and pipe output to unique files
for i in {1..4}
do
    # Create a unique filename using the current timestamp and iteration number
    output_file="catanatron_run_${i}_$(date +%s).log"

    echo "Running game $i, output will be saved to $output_file"

    # Run the command and pipe the output
    catanatron-play --num=500 --code=dqn_player.py --players=DQN,R,R,R --config-map=TOURNAMENT > "$output_file"

    # Check if the command succeeded
    if [ $? -ne 0 ]; then
        echo "Error occurred during run $i. Exiting script."
        exit 1
    fi
done

echo "All runs completed successfully."

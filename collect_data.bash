#!/bin/bash

################################################################################
# Script to collect CARLA-KITTI data for a period of time
#
# Generated with Gemini 3 Pro
################################################################################

# Validate Input
if [ -z "$1" ]; then
    echo "Usage: bash collect_data.bash <number of seconds>"
    exit 1
fi

DURATION_SECONDS=$1
START_TIME=$(date +%s)
END_TIME=$((START_TIME + DURATION_SECONDS))

# Helper function to forcefully kill Carla and wait for it to die
kill_carla() {
    # Loop while the process still exists
    while pgrep -f CarlaUE4 > /dev/null; do
        echo "Attempting to close CARLA simulator..."
        pkill -f CarlaUE4
        sleep 1
    done
}

# Ensure we clean up properly on Ctrl+C
trap "kill_carla; exit" SIGINT SIGTERM

# Initialize the trace counter
TRACE_COUNT=0
echo "Starting data collection loop for $DURATION_SECONDS seconds..."

# Loop until the time has elapsed
while [ $(date +%s) -lt $END_TIME ]; do
    # Format the counter to 4 digits
    printf -v PADDED_COUNT "%04d" $TRACE_COUNT
    OUTPUT_DIR="data/data${PADDED_COUNT}/object"

    echo "------------------------------------------------"
    echo "Starting trace #$PADDED_COUNT -> Output: $OUTPUT_DIR"

    # Start the simulator in the background
    bash $CARLA_ROOT/CarlaUE4.sh -benchmark -fps=10 -quality-level=Low -opengl &

    # Wait for CARLA to initialize
    sleep 2

    # Run the data collector with the new output
    python -m data_collector --loop --save_data --steps_between_recordings=1 --num_frames=200 --output="$OUTPUT_DIR"

    echo "Trace finished. Cleaning up..."
    kill_carla

    # Increment the counter for the next loop
    TRACE_COUNT=$((TRACE_COUNT + 1))
done

echo "Time limit reached. Collection complete."

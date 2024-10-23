#!/bin/bash

################################################################################
# Experiment 01:    TFE25-462
# Date:             22, 23, 24 october 2024
# Objective:        Determine the time taken by each block of the OFDM reciever
#                   to process the data.
# Script:           This script extract the time taken by each function by
#                   running the usrp_rx.py python script. It ouput the results
#                   into a `results.csv` file that can be read by the plotting
#                   script.
#################################################################################

# Varibale definiton
MEASURE_SCRIPT="../src/baseline/usrp_rx.py"
SUB_FOLDERS=("256" "512" "1024" "2048")
RESULTS_FILE="results.csv"
ITERATIONS=5

# Create the results file
echo "function,time,size,iteration" > ${RESULTS_FILE}

# Loop over the subfolders
for folder in ${SUB_FOLDERS[@]}; do
    # Loop over the iterations
    for i in $(seq 1 ${ITERATIONS}); do
        # Run the script
        echo "Running the script for size ${folder}, iteration ${i}"
        folder_arg="${folder}/"
        rx_signal_name="rx_sig_${folder}_rx.dat"
        python3 ${MEASURE_SCRIPT} ${folder_arg} ${rx_signal_name} | sed "s/$/,${folder},${i}/" >> ${RESULTS_FILE}
    done
done

echo "Done"
exit 0

#!/bin/bash
################################################################################
# Experiment 01:    TFE25-462
# Date:             21 november 2024
# Objective:        Realize a timing graph (stacked bar chart) for the poster
#################################################################################

# Varibale definiton
MEASURE_SCRIPT="../../src/baseline/usrp_rx.py"
SUB_FOLDER="2048"
RESULTS_FILE="results_poster.csv"
ITERATIONS=5

# Create the results file
echo "function,time,iteration" > ${RESULTS_FILE}

# Loop over the iterations
for i in $(seq 1 ${ITERATIONS}); do
    # Run the script
    echo "Running the script for iteration ${i}"
    folder_arg="${SUB_FOLDER}/"
    rx_signal_name="rx_sig_${SUB_FOLDER}_rx.dat"
    python3 ${MEASURE_SCRIPT} ${folder_arg} ${rx_signal_name} | sed "s/$/,${i}/" >> ${RESULTS_FILE}
done
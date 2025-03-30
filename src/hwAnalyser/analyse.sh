#!/bin/bash

# VCD files to be analyzed
VCD_DIR="../../gtkwave"
VCD_FILES=(
    "$VCD_DIR/rfnoc_block_schmidl_cox_tb_clipping_large_windows.vcd"
    "$VCD_DIR/rfnoc_block_schmidl_cox_tb_clipping_small_windows.vcd"
    "$VCD_DIR/rfnoc_block_schmidl_cox_tb_truncate_large_windows.vcd"
    "$VCD_DIR/rfnoc_block_schmidl_cox_tb_truncate_small_windows.vcd"
)

# Signals to be analyzed
SIGNALS=(
    "complex rfnoc_block_schmidl_cox_tb.dut.mc0.c4_tdata[63:0] rfnoc_block_schmidl_cox_tb.dut.mc0.c4_tvalid"
    "complex rfnoc_block_schmidl_cox_tb.dut.mc0.c6_tdata[31:0] rfnoc_block_schmidl_cox_tb.dut.mc0.c6_tvalid"
    "real rfnoc_block_schmidl_cox_tb.dut.mc0.r12_tdata[31:0] rfnoc_block_schmidl_cox_tb.dut.mc0.r12_tvalid"
)

# Script to analyze the signals
ANALYZE_SCRIPT="combined_analysis.py"

# For each VCD file, analyze all the signals and save the results in separate directories for each signal
# The ANALYZE_SCRIPT store the result as a PDF file with the signal name in the directory where the script is executed
for VCD_FILE in "${VCD_FILES[@]}"; do
    # Create a directory for the VCD file
    VCD_DIR_NAME=$(basename "$VCD_FILE" .vcd)
    mkdir -p "$VCD_DIR_NAME"

    for SIGNAL in "${SIGNALS[@]}"; do
        # Extract the signal type, tdata name, and tvalid name
        SIGNAL_NAME=$(echo "$SIGNAL" | awk '{print $1}')
        SIGNAL_TDATA=$(echo "$SIGNAL" | awk '{print $2}')
        SIGNAL_TVALID=$(echo "$SIGNAL" | awk '{print $3}')

        # Log the signal information
        echo "Analyzing signal: $SIGNAL_TDATA in $VCD_FILE ..."

        # Run the analysis script with the VCD file and signal names as arguments
        python3 "$ANALYZE_SCRIPT" --vcd_file "$VCD_FILE" --signal_tdata_name "$SIGNAL_TDATA" --signal_tvalid_name "$SIGNAL_TVALID" --signal_type "$SIGNAL_NAME"

        # Move the generated PDF file to the VCD directory
        PDF_FILE=$(ls *.pdf)
        if [ -f "$PDF_FILE" ]; then
            mv "$PDF_FILE" "$VCD_DIR_NAME/$PDF_FILE"
        else
            echo "Error: No PDF file generated for $SIGNAL_NAME in $VCD_FILE"
        fi
    done
done
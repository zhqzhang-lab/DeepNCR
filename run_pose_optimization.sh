#!/bin/bash

# ==============================================================================
# Script Name: run_batch_opt.sh
# Description: Sequentially executes ligand pose optimization for a list of targets.
# Usage:       nohup bash run_batch_opt.sh inputs.dat > opt_log.txt 2>&1 &
# ==============================================================================

INPUT_FILE=$1

# 1. Validate Input
if [ -z "$INPUT_FILE" ]; then
    echo "Error: No input file provided."
    echo "Usage: bash $0 <inputs.dat>"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found."
    exit 1
fi

echo "Starting batch optimization tasks from: $INPUT_FILE"
echo "---------------------------------------------------"

# 2. Process Line by Line
while IFS= read -r line || [ -n "$line" ]
do
    # Skip empty lines
    [[ -z "$line" ]] && continue
    
    # Skip lines starting with # (comments)
    [[ "$line" =~ ^#.*$ ]] && continue

    # Split line into array (space-delimited)
    STR=($line)

    CODE=${STR[0]}
    RECEPTOR_PATH=${STR[1]}
    POSES_PATH=${STR[2]}

    OUTPUT_PATH="Posebusters_opt_output_2/${CODE}"

    # 3. Check for Existing Output (Resume Capability)
    if [ -d "$OUTPUT_PATH" ]; then
        echo "[INFO] Skipping ${CODE}, output directory exists: ${OUTPUT_PATH}"
        continue
    fi

    # Create directory
    mkdir -p "$OUTPUT_PATH"

    # Log current task
    echo "============================================================"
    echo "▶ Processing Target: ${CODE}"
    echo "  Receptor: ${RECEPTOR_PATH}"
    echo "  Poses:    ${POSES_PATH}"
    echo "  Output:   ${OUTPUT_PATH}"
    echo "============================================================"

    # 4. Execute Python Optimization Script
    # Note: Running sequentially to ensure GPU memory safety and avoid conflicts.
    python scripts_transformer/run_opt.py \
        --receptor "$RECEPTOR_PATH" \
        --poses "$POSES_PATH" \
        --output "$OUTPUT_PATH"

    # Check execution status
    if [ $? -eq 0 ]; then
        echo "✔ [Success] ${CODE} processing done."
    else
        echo "✘ [Error] Failed to process ${CODE}. Check logs for details."
    fi
    echo ""

done < "$INPUT_FILE"

echo "🎉 All tasks completed!"
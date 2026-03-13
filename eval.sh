#!/bin/bash

BASE_DIR="/data2/mnt2/zhoudw/zh/tangjt/PyMCIT"
SCRIPTS_DIR="$BASE_DIR/scripts/SAME/Eval_CoIN"
OUTPUT_DIR="/data2/mnt2/zhoudw/zh/tangjt/PyMCIT/SAME_output/test_last"

mkdir -p "$OUTPUT_DIR"

for i in {1..8}; do
    script_path=$(find "$SCRIPTS_DIR" -name "${i}_*.sh" | head -n 1)
    if [ -z "$script_path" ]; then
        echo "❌ script ${i}_*.sh not FOUND!"
        continue
    fi
    output_file="$OUTPUT_DIR/last_${i}.txt"

    echo "🔄 starting task $i..."
    bash "$script_path" > "$output_file" 2>&1
    echo "✅ task $i completed -> $output_file"
done

echo "🎉 test completed!"
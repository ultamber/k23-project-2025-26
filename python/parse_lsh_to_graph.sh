#!/bin/bash
# parse_lsh_to_graph.sh
# Quick script to parse LSH output and use it for index building

LSH_OUTPUT="lsh_output.txt"
KNN_GRAPH="mnist_knn_k10.npy"
N_QUERIES=60000
K_NEIGHBORS=10

echo "Step 1: Parsing LSH output..."
python parse_lsh_output.py \
    -i "$LSH_OUTPUT" \
    -o "$KNN_GRAPH" \
    -n "$N_QUERIES" \
    -k "$K_NEIGHBORS"

echo ""
echo "Step 2: Building Neural LSH index..."
python nlsh_build.py \
    -d ../datasets/MNIST/train-images.idx3-ubyte \
    -i index_from_lsh \
    -type mnist \
    -m 50 \
    --knn_graph_file "$KNN_GRAPH" \
    --layers 3 \
    --nodes 128 \
    --dropout 0.0 \
    --epochs 20 \
    --batch_size 256

echo ""
echo "✓ Done! Index built using LSH k-NN graph"
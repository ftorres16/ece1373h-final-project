#!/bin/bash

# Run inside the hls folder

echo "Compiling..."

make all

echo ""
echo "Testing..."

./bin/batch_norm_2d
./bin/conv
./bin/fc
./bin/max_pool_2d
./bin/relu
./bin/conv_relu
./bin/conv_batch_relu
./bin/conv_batch_relu_max
./bin/full_nn

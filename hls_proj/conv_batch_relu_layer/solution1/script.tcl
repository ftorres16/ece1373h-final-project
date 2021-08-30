############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
open_project conv_batch_relu_layer
set_top conv_batch_relu_layer
add_files ../hls/src/layers/conv_batch_relu.cpp
add_files ../hls/src/layers/conv_batch_relu.h
add_files ../hls/src/layers/conv.cpp
add_files ../hls/src/layers/conv.h
add_files -tb ../hls/tb_data/conv_batch_relu.txt
add_files -tb ../hls/tb/tb_config.h
add_files -tb ../hls/tb/layers/test_conv_batch_relu.cpp
add_files -tb ../hls/tb/utils.cpp
add_files -tb ../hls/tb/utils.h
open_solution "solution1"
set_part {xc7z030sbg485-1} -tool vivado
create_clock -period 10 -name default
source "./conv_batch_relu_layer/solution1/directives.tcl"
csim_design
csynth_design
cosim_design -trace_level all
export_design -format ip_catalog

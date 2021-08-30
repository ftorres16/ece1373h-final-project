############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
open_project conv_batch_relu_max_layer
set_top conv_batch_relu_max_layer
add_files ../hls/src/layers/conv_batch_relu_max.cpp
add_files ../hls/src/layers/conv_batch_relu_max.h
add_files ../hls/src/layers/conv_batch_relu.cpp
add_files ../hls/src/layers/conv_batch_relu.h
add_files ../hls/src/layers/conv.cpp
add_files ../hls/src/layers/conv.h
add_files ../hls/src/layers/max_pool_2d.cpp
add_files ../hls/src/layers/max_pool_2d.h
add_files -tb ../hls/tb_data/conv_batch_relu_max.txt -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb ../hls/tb_data/conv_batch_relu_max_pre.txt -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb ../hls/tb_data/conv_batch_relu_max_params.txt -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb ../hls/tb/tb_config.h -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb ../hls/tb/layers/test_conv_batch_relu_max.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb ../hls/tb/utils.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb ../hls/tb/utils.h -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution1"
set_part {xc7z030-sbg485-1} -tool vivado
create_clock -period 10 -name default
config_export -format ip_catalog
source "./conv_batch_relu_max_layer/solution1/directives.tcl"
csim_design
csynth_design
cosim_design -reduce_diskspace -trace_level port
export_design -format ip_catalog

############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
open_project bar
set_top bar
add_files ../hls/src/bar/bar.cpp
add_files ../hls/src/bar/bar.h
add_files ../hls/src/layers/conv.cpp
add_files ../hls/src/layers/conv.h
add_files ../hls/src/layers/conv_batch_relu.cpp
add_files ../hls/src/layers/conv_batch_relu.h
add_files ../hls/src/layers/conv_batch_relu_max.cpp
add_files ../hls/src/layers/conv_batch_relu_max.h
add_files ../hls/src/layers/conv_relu.cpp
add_files ../hls/src/layers/conv_relu.h
add_files ../hls/src/layers/max_pool_2d.cpp
add_files ../hls/src/layers/max_pool_2d.h
add_files ../hls/src/layers/zero_mean.cpp
add_files ../hls/src/layers/zero_mean.h
add_files -tb ../hls/tb/utils.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb ../hls/tb/utils.h -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb ../hls/tb/tb_config.h -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb ../hls/tb/bar/test_bar.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb ../hls/tb_data/bar.txt -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb ../hls/tb_data/bar_pre.txt -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution1"
set_part {xc7z030-sbg485-1} -tool vivado
create_clock -period 10 -name default
config_export -format ip_catalog
source "./bar/solution1/directives.tcl"
csim_design
csynth_design
cosim_design -setup -reduce_diskspace -trace_level port
export_design -format ip_catalog

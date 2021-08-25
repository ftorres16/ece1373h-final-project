############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
open_project zero_mean_layer
set_top zero_mean_layer
add_files ../hls/src/layers/zero_mean.cpp
add_files ../hls/src/layers/zero_mean.h
add_files -tb ../hls/tb_data/zero_mean.txt -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb ../hls/tb_data/zero_mean_params.txt -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb ../hls/tb/tb_config.h -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb ../hls/tb/layers/test_zero_mean.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb ../hls/tb/utils.cpp -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
add_files -tb ../hls/tb/utils.h -cflags "-Wno-unknown-pragmas" -csimflags "-Wno-unknown-pragmas"
open_solution "solution1"
set_part {xc7z030-sbg485-1} -tool vivado
create_clock -period 10 -name default
config_export -format ip_catalog
source "./zero_mean_layer/solution1/directives.tcl"
csim_design
csynth_design
cosim_design -reduce_diskspace -trace_level port
export_design -format ip_catalog

############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
open_project max_pool_2d_layer
set_top max_pool_2d
add_files ../hls/src/layers/max_pool_2d.cpp
add_files ../hls/src/layers/max_pool_2d.h
add_files -tb ../hls/tb_data/max_pool_2d.txt
add_files -tb ../hls/tb_data/max_pool_2d_params.txt
add_files -tb ../hls/tb/tb_config.h
add_files -tb ../hls/tb/layers/test_max_pool_2d.cpp
add_files -tb ../hls/tb/utils.cpp
add_files -tb ../hls/tb/utils.h
open_solution "solution1"
set_part {xc7z030sbg485-1} -tool vivado
create_clock -period 10 -name default
source "./max_pool_2d_layer/solution1/directives.tcl"
csim_design
csynth_design
cosim_design -trace_level all
export_design -format ip_catalog

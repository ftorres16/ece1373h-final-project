############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
open_project spike_deeptector_cumulative
set_top spike_deeptector_cumulative
add_files ../hls/src/spike_deeptector/spike_deeptector_cumulative.cpp
add_files ../hls/src/spike_deeptector/spike_deeptector_cumulative.h
add_files -tb ../hls/tb/spike_deeptector/test_spike_deeptector_cumulative.cpp
open_solution "solution1"
set_part {xc7z030-sbg485-1} -tool vivado
create_clock -period 10 -name default
config_export -format ip_catalog
source "./spike_deeptector_cumulative/solution1/directives.tcl"
csim_design
csynth_design
cosim_design -setup -reduce_diskspace -trace_level port
export_design -format ip_catalog

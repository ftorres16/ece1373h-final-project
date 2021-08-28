############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2019 Xilinx, Inc. All Rights Reserved.
############################################################
set_directive_interface -mode s_axilite -bundle CTRL_BUS "bar_main"
set_directive_interface -mode m_axi -depth 1929600 "bar_main" mem
set_directive_interface -mode s_axilite -bundle CTRL_BUS "bar_main" mem_params
set_directive_interface -mode s_axilite -bundle CTRL_BUS "bar_main" n_samples
set_directive_interface -mode m_axi -depth 48 "bar_main" samples_offset
set_directive_interface -mode s_axilite -bundle CTRL_BUS "bar_main" n_spikes
set_directive_interface -mode m_axi -depth 48 "bar_main" spikes_offset

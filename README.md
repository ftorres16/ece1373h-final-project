# ece1373h-final-project

## Setup

> :warning: Disclaimer: Scripts are mean to be run with Vivado Suite version 2019.1, use other versions at your own risk.

In order to access PicoZed FMC Carrier Card 2 as a device / board from Vivado and Vivado HLS, 
copy the 'picozed_7030_fmc2' directory to the following path:
<Xilinx_install_dir>\Vivado\2019.1\data\boards\board_files

## Project organization

This repository contains the following relevant folders:

### hls

Contains the C++ source code to be synthesized by Vivado HLS.

### hls_proj

Contains the Vivado HLS projects for implementing the hardware.

### tb_gen_data

Python project for generating testbench data to be used in the C++ testbenches.

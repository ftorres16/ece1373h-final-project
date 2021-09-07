cd <Path to Vivado .sdk directory/*_hw_platform_0>
connect
target 2
rst
fpga <name_of .bit file>
loadhw system.hdf
source ps7_init.tcl
ps7_init
ps7_post_config
dow < Path to Debug/name_of.elf>
# con

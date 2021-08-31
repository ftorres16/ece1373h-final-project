# Generate tb data

PyTorch implementation to generate validation data for the deep learning layers to be implemented in HLS.

## Requirements

1. Python 3.8+
2. [Python Poetry](python-poetry.org/)

## Installation

1. `cd` into the folder where this file is.
2. Run `poetry install`
3. You can access all the funcionality from an interactive shell after running `poetry shell`.

## Generating testbench data

1. `cd` into the folder where this file is.
2. Run `poetry run python tb_gen_data/gen_<block>.py`, where `<block>` is the block you want to generate data for. For example: `poetry run python tb_gen_data/gen_spike_deepclassifier.py`
3. Output data will be generated in the folder `./outputs/`, generally in `.txt` format. For the top block, a `.bin` file is also generated that can be used as a memory image.

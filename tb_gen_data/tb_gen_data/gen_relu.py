import torch
import torch.nn as nn

from config import BATCH_SIZE, IN_X, IN_Y, OUTPUT_FOLDER


def create_random_data(out_file: str):
    relu = nn.ReLU()

    input_ = torch.randn(BATCH_SIZE, IN_X, IN_Y)
    output = relu(input_)

    with open(out_file, "w") as f:
        in_str = [f"{in_px}\n" for in_px in torch.flatten(input_).tolist()]
        f.writelines(in_str)

        out_str = [f"{out_px}\n" for out_px in torch.flatten(output).tolist()]
        f.writelines(out_str)


if __name__ == "__main__":
    create_random_data(f"{OUTPUT_FOLDER}/relu.txt")

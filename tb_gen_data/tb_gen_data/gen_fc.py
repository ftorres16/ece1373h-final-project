import torch

from config import BATCH_SIZE, IN_X, IN_Y, OUTPUT_FOLDER
from tb_gen_data.layers.fc import FC


def create_random_data(out_file: str):
    fc = FC()

    input_ = torch.randn(BATCH_SIZE, IN_X, IN_Y)
    output = fc(input_)

    with open(out_file, "w") as f:
        weights = [f"{weight}\n" for weight in torch.flatten(fc.linear.weight).tolist()]
        f.writelines(weights)

        biases = [f"{bias}\n" for bias in fc.linear.bias.tolist()]
        f.writelines(biases)

        in_str = [f"{in_px}\n" for in_px in torch.flatten(input_).tolist()]
        f.writelines(in_str)

        out_str = [f"{out_px}\n" for out_px in torch.flatten(output).tolist()]
        f.writelines(out_str)


if __name__ == "__main__":
    create_random_data(f"{OUTPUT_FOLDER}/fc.txt")

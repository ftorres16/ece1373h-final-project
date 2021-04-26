import torch

from layers.cnn import CNN
from config import BATCH_SIZE, IN_D, IN_Y, IN_X, OUTPUT_FOLDER

torch.manual_seed(0)


def create_random_data(out_file: str):
    cnn = CNN()

    input_ = torch.randn(BATCH_SIZE, IN_D, IN_Y, IN_X)
    output = cnn(input_)

    with open(out_file, "w") as f:
        weights = [f"{weight}\n" for weight in torch.flatten(cnn.conv.weight).tolist()]
        f.writelines(weights)

        biases = [f"{bias}\n" for bias in cnn.conv.bias.tolist()]
        f.writelines(biases)

        in_str = [f"{in_px}\n" for in_px in torch.flatten(input_).tolist()]
        f.writelines(in_str)

        out_str = [f"{out_px}\n" for out_px in torch.flatten(output).tolist()]
        f.writelines(out_str)


if __name__ == "__main__":
    create_random_data(f"{OUTPUT_FOLDER}/cnn.txt")

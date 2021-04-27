import torch

from tb_gen_data.gen_batch_norm import GenBatchNorm2D
from tb_gen_data.gen_cnn import GenCNN
from tb_gen_data.gen_fc import GenFC
from tb_gen_data.gen_max_pool_2d import GenMaxPool2D
from tb_gen_data.gen_relu import GenReLU

if __name__ == "__main__":
    gens = {
        GenBatchNorm2D: "batch_norm_2d",
        GenCNN: "cnn",
        GenFC: "fc",
        GenMaxPool2D: "max_pool_2d",
        GenReLU: "relu",
    }

    for gen_, name in gens.items():
        torch.manual_seed(0)

        gen = gen_(name)

        gen.gen_input()
        gen.gen_output()
        gen.write_mem()
        gen.write_params()

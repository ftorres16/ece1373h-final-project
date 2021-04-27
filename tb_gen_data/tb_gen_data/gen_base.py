import typing as T

from tb_gen_data.config import OUTPUT_FOLDER


class GenBase:
    def __init__(self, name: str, out_folder: str = OUTPUT_FOLDER):
        self.name = name
        self.out_folder = out_folder

        self.out_file = f"{out_folder}/{name}.txt"
        self.params_file = f"{out_folder}/{name}_params.txt"

        self.input_ = None
        self.output = None

        self.mem: T.Dict[str, int] = []
        self.params: T.List[str] = {}

    def gen_input(self):
        """
        Generate random pattern and store it in `self.input_`.
        """
        raise NotImplementedError("This is an abstract class.")

    def gen_output(self):
        """
        Generate output of the neural net and store it in `self.output`.
        """
        raise NotImplementedError("This is an abstract class.")

    def _gen_mem(self):
        """
        Load the resulting memory layout from running the module in `self.mem`.
        """
        raise NotImplementedError("This is an abstract class.")

    def write_mem(self):
        """
        Write the memory layout from `self.mem` to a text file in path `self.out_file`.
        """
        self._gen_mem()

        with open(self.out_file, "w") as f:
            f.writelines(self.mem)

    def _gen_params(self):
        """
        Load the module parameters in `self.params`.
        """
        raise NotImplementedError("This is an abstract class.")

    def write_params(self):
        """
        Write the contents of  `self.params` to a text file in path `self.params_file`.
        """
        self._gen_params()

        with open(self.params_file, "w") as f:
            for key, val in self.params.items():
                f.write(f"{key}\n")
                f.write(f"{val}\n")

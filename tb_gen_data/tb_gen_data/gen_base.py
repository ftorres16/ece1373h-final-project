class GenBase:
    def __init__(self, out_file: str):
        self.out_file = out_file

        self.input_ = None
        self.output = None

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

    def write_output(self):
        """
        Write the output to a text file in path `out_file`.
        """
        raise NotImplementedError("This is an abstract class.")

import numpy as np


class TensorLogger:

    def __init__(self, name=None):
        self.name = name if name is not None else f"tensor_log_{np.random.randint(99999999):08}"
        self.tensors = []
        self.size_limit = 1000
        self.cur_file = 0

    def __del__(self):
        self.save()
        self.tensors = []

    def log(self, image):
        self.tensors.append(image)
        if len(self.tensors) >= self.size_limit:
            self.save()

    def save(self):
        np.save(f"data/records/{self.name}.{self.cur_file:03}.npy", self.tensors)
        self.tensors = []
        self.cur_file += 1

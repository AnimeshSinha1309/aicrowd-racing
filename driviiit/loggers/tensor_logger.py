import typing as ty

import numpy as np


class TensorLogger:
    def __init__(self, name=None, fields: ty.Tuple[str, ...] = ('tensor',)):
        self.name = name if name is not None else f"tensor_log_{np.random.randint(99999):08}"
        self.fields = fields
        self.tensors = {field: [] for field in fields}
        self.size_limit = 100000
        self.cur_file = 0

    def __del__(self):
        self.save()

    def log(self, **kwargs):
        for key, value in kwargs.items():
            self.tensors[key].append(value)
        if len(self.tensors) >= self.size_limit:
            self.save()

    def save(self):
        for key in self.tensors.keys():
            self.tensors[key] = np.stack(self.tensors[key], axis=0)
        np.savez(f"data/records/{self.name}.{self.cur_file:03}", **self.tensors)
        self.tensors = {field: [] for field in self.fields}
        self.cur_file += 1

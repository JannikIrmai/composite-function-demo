import numpy as np
from abc import ABC, abstractmethod


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.max(0, x)


def identity(x):
    return x


class Function(ABC):

    def __init__(self, params: dict = None):
        self._params = params if params is not None else {}

    @abstractmethod
    def forward(self, *args):
        pass


class ConstantFunction(Function):

    def __init__(self, params: dict, param_idx):
        super().__init__(params)
        self._param_idx = param_idx

    def forward(self):
        return self._params.get(self._param_idx, 1.0)


class ConstantOne(Function):

    def __init__(self, params: dict):
        super().__init__(params)

    def forward(self):
        return 1.0


class IdentityFunction(Function):

    def forward(self, arg):
        return arg


class MLP(Function):

    def __init__(self, params: dict = None):
        super().__init__(params)
        self._param_ids = []
        self._activation = identity

    def set_activation(self, activation):
        self._activation = activation

    def add_param(self, param):
        self._param_ids.append(param)

    def remove_param(self, param):
        self._param_ids.remove(param)

    def forward(self, *args):
        assert len(args) == len(self._param_ids)
        if len(args) == 0:
            return self._activation(0)
        weighted_sum = 0
        for arg, p in zip(args, self._param_ids):
            weighted_sum = weighted_sum + arg * self._params.get(p, 1)
        return self._activation(weighted_sum)





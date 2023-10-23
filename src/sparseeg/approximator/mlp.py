from flax import linen as nn
from typing import Sequence, Callable


class MLP(nn.Module):
    features: Sequence[int]
    act: Sequence[Callable]
    weight_init: Callable

    def setup(self):
        self.layers = [
            nn.Dense(feat, kernel_init=self.weight_init)
            for feat in self.features
        ]

    def __call__(self, input_):
        x = input_
        for i, l in enumerate(self.layers):
            x = l(x)
            x = self.act[i](x)
        return x

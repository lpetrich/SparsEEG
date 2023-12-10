"""
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License. 
"""

from pprint import pprint
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

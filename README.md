# GOPT

# Introduction

GOPT is a framework to run global optimization meta-euristics on combinatorial problems (TSP, Bin-Packing, etc...)

The key charactirstics are:

- **Ease of development**: Problems, algorithms, and even the framework are written in Python
- **Performance**: Before running, all the code is transpiled using [Numba](https://github.com/numba/numba) to provide native speed
- **Retargetable**: The code should be able to run on various hardware: CPU, Cuda, ROCM, remote clusters etc...
- **Instance Optimized**: The generated code depends on a specific instance of the problem. For example, in the EuclideanTSP case, the number of cities and the dimensionality of the space is known in advanced and the compiler is able to specialize the code (loop unrooling, etc...).

# Concepts

We describe here the main abstractions used by GOPT. Users can use the ones provided in this repository or swap them by their own implementation.

## Problem

## Optimizer

## Shuffler

# Getting Started

# Quick example

```python
import numba
import numpy as np
import logging

from gopt import Compiler
from gopt.problems import EuclieanTSP
from gopt.optimizers import LocalSearch
from gopt.runners import CPURunner
from gopt.shufflers import IndependentShuffler

Compiler.debug = False
# To speed up compilation during development
# Should be 'always' for long running jobs
Compiler.inline = 'never'

# root = logging.getLogger('gopt.compiler.LocalSearch')
logging.basicConfig(format='%(levelname)s [%(name)s]:%(message)s',
                    level=logging.INFO)

DATA = np.random.uniform(size=(1000, 2)).astype('float32')
NUM_CITIES = DATA.shape[0]

TSP = EuclieanTSP(DATA.shape[0], DATA.shape[1])
Optimizer = LocalSearch(TSP)
Shuffler = IndependentShuffler(Optimizer, 32)

runner = CPURunner(Shuffler, DATA)
result = runner.run(max_iter=1000000000, max_time='1 min')
```
# Benchmarks

import numba
import numpy as np
import logging

from gopt import Compiler
from gopt.problems import EuclieanTSP
from gopt.optimizers import LocalSearch
from gopt.runners import SingleCoreCPURunner

Compiler.debug=False

root = logging.getLogger()
root.setLevel(logging.INFO)

DATA = np.random.uniform(size=(1000, 2)).astype('float32')
NUM_CITIES = DATA.shape[0]

TSP = EuclieanTSP(DATA.shape[0], DATA.shape[1])
runner = SingleCoreCPURunner(TSP, LocalSearch, DATA)
loss, solution = runner.run(max_iter=1000000000, max_time='10 sec')
print(solution)

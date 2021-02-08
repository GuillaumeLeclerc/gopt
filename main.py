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
Shuffler = IndependentShuffler(Optimizer, 1000)

runner = CPURunner(Shuffler, DATA, 64)
loss, solution = runner.run(max_iter=100000000, max_time='5 min')
print(loss, solution)

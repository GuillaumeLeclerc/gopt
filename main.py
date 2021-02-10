import numba
import numpy as np
import logging

from gopt import Compiler
from gopt.problems import EuclieanTSP
from gopt.optimizers import RandomLocalSearch
from gopt.runners import CPURunner
from gopt.shufflers import IndependentShuffler, WinnerTakesAll

Compiler.debug = False
# To speed up compilation during development
# Should be 'always' for long running jobs
Compiler.inline = 'never'

# root = logging.getLogger('gopt.compiler.LocalSearch')
logging.basicConfig(format='%(levelname)s [%(name)s]:%(message)s',
                    level=logging.INFO)

DATA = np.random.uniform(size=(10000, 2)).astype('float32')
NUM_CITIES = DATA.shape[0]

TSP = EuclieanTSP(DATA.shape[0], DATA.shape[1],
                  neighborhood='2-opt', init='NN')
Optimizer = RandomLocalSearch(TSP)
Shuffler = WinnerTakesAll(Optimizer, 16)

runner = CPURunner(Shuffler, DATA)
result = runner.run(max_iter=1000000000, max_time='1 min')
print(result)

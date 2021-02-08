import numba
import numpy as np
import logging

from gopt import Compiler
from gopt.problems import EuclieanTSP
from gopt.optimizers import SimulatedAnnealing
from gopt.runners import SingleCoreCPURunner
from gopt.shufflers import IndependentShuffler

Compiler.debug = False 
# To speed up compilation during development
# Should be 'always' in production
Compiler.inline = 'never'

# root = logging.getLogger('gopt.compiler.LocalSearch')
logging.basicConfig(format='%(levelname)s [%(name)s]:%(message)s',
                    level=logging.INFO)

DATA = np.random.uniform(size=(1000, 2)).astype('float32')
NUM_CITIES = DATA.shape[0]
meta_algo = '2-opt'

TSP = EuclieanTSP(DATA.shape[0], DATA.shape[1], meta_algo)
Optimizer = SimulatedAnnealing(TSP, 1000, 1e-4)
Shuffler = IndependentShuffler(Optimizer, 40)
Shuffler.compile()

runner = SingleCoreCPURunner(TSP, Optimizer, DATA)
loss, solution = runner.run(max_iter=1000000000, max_time='10 sec')

import numba
import numpy as np
import logging

from gopt import Compiler
from gopt.problems import EuclieanTSP
from gopt.optimizers import SimulatedAnnealing
from gopt.runners import SingleCoreCPURunner

Compiler.debug = False
# To speed up compilation during development
# Should be 'always' in production
Compiler.inline = 'never'

# root = logging.getLogger('gopt.compiler.LocalSearch')
logging.basicConfig(format='%(levelname)s [%(name)s]:%(message)s',
                    level=logging.INFO)

DATA = np.random.uniform(size=(1000, 2)).astype('float32')
NUM_CITIES = DATA.shape[0]

TSP = EuclieanTSP(DATA.shape[0], DATA.shape[1])
optimizer = SimulatedAnnealing(TSP, 1000, 1e-4)
runner = SingleCoreCPURunner(TSP, optimizer, DATA)
loss, solution = runner.run(max_iter=1000000000, max_time='10 sec')
print(solution)

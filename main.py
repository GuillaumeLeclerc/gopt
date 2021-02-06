import numba
import numpy as np
import logging

from gopt import Problem
from gopt.optimizers import LocalSearch
from gopt.runners import SingleCoreCPURunner

root = logging.getLogger()
root.setLevel(logging.INFO)

DATA = np.random.uniform(size=(1000, 2))
NUM_CITIES = DATA.shape[0]


@numba.njit()
def distance(a, b, problem_data):
    result = (problem_data[a, 0] - problem_data[b, 0]) ** 2
    result += (problem_data[a, 1] - problem_data[b, 1]) ** 2
    return result


class TSP(Problem):
    problem_name = 'TSP'
    state_dtype = np.dtype([
        ('order', (np.int32, NUM_CITIES))
    ])
    problem_data_dtype = np.dtype((DATA.dtype, DATA.shape))

    def state_init(state, problem_data):
        for i in range(NUM_CITIES):
            state['order'][i] = i

    def loss(state_array, problem_data):
        result = 0
        order = state_array['order']
        for i in range(NUM_CITIES - 1):
            result += distance(order[i], order[i + 1], problem_data)
        result += distance(order[0], order[NUM_CITIES - 1], problem_data)

        return result

    def neighbor(state, problem_data):
        a = np.random.randint(0, NUM_CITIES)
        b = np.random.randint(0, NUM_CITIES - 1)
        if b >= a:
            b += 1
        order = state['order']
        order[a], order[b] = order[b], order[a]


runner = SingleCoreCPURunner(TSP, LocalSearch, DATA)
loss, solution = runner.run(max_iter=1000000000, max_time='10 sec')
print(solution)

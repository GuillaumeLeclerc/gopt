import logging
import numba
import numpy as np
from ..compiler import Compiler

from .. import Problem

def EuclieanTSP(num_cities, dimensionality, dtype=np.float32):
    if dimensionality < 2:
        logging.warning("Seriously ? -_-")

    @Compiler.ufunc
    def distance(a, b, problem_data):
        result = 0.0
        for i in range(dimensionality):
            result += (problem_data[a, i] - problem_data[b, i]) ** 2
        return result

    class TSP(Problem):
        problem_name = 'TSP'
        state_dtype = np.dtype([
            ('order', (np.int32, num_cities))
        ])
        problem_data_dtype = np.dtype((dtype, (num_cities, dimensionality)))

        @staticmethod
        def state_init(state, problem_data):
            for i in range(num_cities):
                state['order'][i] = i

        @staticmethod
        def loss(state_array, problem_data):
            result = 0
            order = state_array['order']
            for i in range(num_cities - 1):
                result += distance(order[i], order[i + 1], problem_data)
            result += distance(order[0], order[num_cities - 1], problem_data)

            return result

        @staticmethod
        def neighbor(state, problem_data):
            a = np.random.randint(0, num_cities)
            b = np.random.randint(0, num_cities - 1)
            if b >= a:
                b += 1
            order = state['order']
            order[a], order[b] = order[b], order[a]

    return TSP

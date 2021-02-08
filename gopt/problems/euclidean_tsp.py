import logging
import numba
import numpy as np
from ..compiler import Compiler

from .. import Problem

def EuclieanTSP(num_cities, dimensionality, meta_algo, dtype=np.float32):
    if dimensionality < 2:
        logging.warning("Seriously ? -_-")

    @Compiler.ufunc
    def distance(a, b, problem_data):
        result = 0.0
        for i in range(dimensionality):
            result += (problem_data[a, i] - problem_data[b, i]) ** 2
        return result

    @Compiler.ufunc
    def neighbor_localS(state, problem_data):
        order = state['order']
        def d(a, b):
            return distance(order[f(a)], order[f(b)], problem_data)

        a = np.random.randint(0, num_cities)
        b = np.random.randint(0, num_cities - 1)
        if b >= a:
            b += 1

        loss_diff = 0
        loss_diff -= d(a - 1, a) + d(a, a + 1) + d(b - 1, b) + d(b, b + 1)
        order[a], order[b] = order[b], order[a]
        loss_diff += d(a - 1, a) + d(a, a + 1) + d(b - 1, b) + d(b, b + 1)

        return loss_diff

    @Compiler.ufunc
    def neighbor_twoOpt(state, problem_data):
        order = state['order']
        def d(a, b):
            return distance(order[f(a)], order[f(b)], problem_data)

        a = np.random.randint(0, num_cities - 1)
        c = np.random.randint(0, num_cities - 2)
           
        b, e = a+1, c+1
        loss_diff = 0
        loss_diff -= d(a, b) + d(c, e)
        order[b], order[e] = order[e], order[b]
        loss_diff += d(a, c) + d(b, e)
        return loss_diff
        

    if meta_algo == 'localSearch':
        neighbor_func = neighbor_localS
    elif meta_algo == '2-opt':
        neighbor_func = neighbor_twoOpt

    @Compiler.ufunc
    def f(i):
        return i % num_cities

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
            for i in range(num_cities):
                result += distance(order[f(i)], order[f(i + 1)], problem_data)
            return result

        @staticmethod
        def neighbor(state, problem_data):
            return neighbor_func(state, problem_data)

    return TSP

import logging
import numba
import numpy as np
from ..compiler import Compiler

from .. import Problem

def EuclieanTSP(num_cities, dimensionality, meta_algo='localS', init='NN', dtype=np.float32):
    if dimensionality < 2:
        logging.warning("Seriously ? -_-")

    @Compiler.ufunc
    def distance(a, b, problem_data):
        result = 0.0
        for i in range(dimensionality):
            result += (problem_data[a, i] - problem_data[b, i]) ** 2
        return result

    @Compiler.ufunc
    def f(i):
        return i % num_cities

    @Compiler.ufunc
    def random_init(state, problem_data):
        for i in range(num_cities):
            state['order'][i] = i
        
    @Compiler.ufunc

    def NN_init(state, problem_data):
        unvisited = np.arange(num_cities)
        curr_node = np.random.choice(unvisited)
        unvisited = np.delete(unvisited, curr_node)
        state['order'][0] = curr_node
        i = 0
        while i < num_cities - 1:
            curr_node = int(state['order'][i])
            best_node, best_idx = unvisited[0], 0
            min_dist = distance(curr_node, best_node, problem_data)
            for j in range(len(unvisited)):
                node = unvisited[j]
                dist =  distance(curr_node, node, problem_data)
                if dist < min_dist:
                    best_node = node
                    best_idx = j
                    min_dist = dist
            i+=1
            unvisited = np.delete(unvisited, best_idx)
            state['order'][i] = best_node
        return state

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

        a = np.random.randint(0, num_cities-4)
        c = np.random.randint(a+2, num_cities-2)
        
        b, e = a+1, c+1
        loss_diff = d(a, c) + d(b, e) - d(a, b) + d(c, e) 
        order[b:c+1] = order[b:c+1][::-1]
        return loss_diff

    init_func = random_init if init=='random' else NN_init

    if meta_algo == 'localSearch':
        neighbor_func = neighbor_localS
    elif meta_algo == '2-opt':
        neighbor_func = neighbor_twoOpt
    else:
        raise ValueError#(f'{meta_algo} not available, choose from \"localSearch"/\"\2-opt"')


    class TSP(Problem):
        problem_name = 'TSP'
        state_dtype = np.dtype([
            ('order', (np.int32, num_cities))
        ])
        problem_data_dtype = np.dtype((dtype, (num_cities, dimensionality)))

        @staticmethod
        def state_init(state, problem_data):
            init_func(state, problem_data)

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

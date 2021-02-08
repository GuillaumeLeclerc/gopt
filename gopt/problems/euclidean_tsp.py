import logging
from os import stat
import numba
import numpy as np
from ..compiler import Compiler

from .. import Problem

def EuclieanTSP(num_cities, dimensionality, meta_algo='localSearch', init='NN', dtype=np.float32):
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

    def random_init(state, problem_data):
        for i in range(num_cities):
            state['order'][i] = i
        
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
    
    def neighbor_twoOpt(state, problem_data):
        order = state['order']
        def d(a, b):
            return distance(order[f(a)], order[f(b)], problem_data)

        loss_diff_tot = 0
        for i in range(num_cities-2):
            for j in range(i+1, num_cities-1):
                loss_diff = 0
                loss_diff -= d(i, i+1) + d(j, j+1) 
                loss_diff += d(i, j) + d(i+1, j+1) 
                if loss_diff < 0:
                    order[i+1:j+1] = order[i+1:j+1][::-1]
                    loss_diff_tot +=  loss_diff
        return loss_diff_tot

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
        def loss(state_array, problem_data):
            result = 0
            order = state_array['order']
            for i in range(num_cities):
                result += distance(order[f(i)], order[f(i + 1)], problem_data)
            return result

    TSP.neighbor = staticmethod(neighbor_func)
    TSP.state_init = staticmethod(init_func)
    return TSP

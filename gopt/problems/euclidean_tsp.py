import logging
from os import stat
import numba
import numpy as np
from ..compiler import Compiler

from .. import Problem

def EuclieanTSP(num_cities, dimensionality, neighborhood='2-opt', init='NN',
                dtype=np.float32):
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

    def random_init(state, _):
        for i in np.random.permutation(np.arange(num_cities)):
            state['order'][i] = i

    def nn_init(state, problem_data):
        unvisited = []
        for i in range(num_cities):
            unvisited.append(i)
        unvisited = np.array(unvisited)
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
                dist = distance(curr_node, node, problem_data)
                if dist < min_dist:
                    best_node = node
                    best_idx = j
                    min_dist = dist
            i += 1
            unvisited = np.delete(unvisited, best_idx)
            state['order'][i] = best_node

    def neighbor_swapTwo(state, problem_data):
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

        a = np.random.randint(0, num_cities)
        b = np.random.randint(0, num_cities - 1)
        if b >= a:
            b += 1
        if b < a:
            a, b = b, a

        # This is just reverting the visiting order of cities
        # No change so we don't do anything
        if a == 0 and b == num_cities - 1:
            return 0.0

        loss_diff = 0
        loss_diff -= d(a - 1, a) + d(b, b + 1)
        loss_diff += d(a - 1, b) + d(a, b + 1)

        for i in range(0, (b - a + 1) // 2):
            order[a + i], order[b - i] = order[b - i], order[a + i]

        return loss_diff

    init_funcs = {
        'random': random_init,
        'NN': nn_init
    }
    neighbor_funcs = {
        'swap-2': neighbor_swapTwo,
        '2-opt': neighbor_twoOpt
    }

    init_func = init_funcs.get(init, None)
    if init_func is None:
        raise ValueError(
            f"{init} not available, choose from {', '.join(init_funcs.keys())}"
        )

    neighbor_func = neighbor_funcs.get(neighborhood, None)
    if neighbor_func is None:
        raise ValueError(f"""{meta_algo} not available, choose from
                         {', '.join(neighbor_funcs.keys())}""")

    class TSP(Problem):
        problem_name = 'TSP'
        state_dtype = np.dtype([
            ('order', (np.int32, num_cities))
        ])
        problem_data_dtype = np.dtype((dtype, (num_cities, dimensionality)))

    if neighbor_func is None:
        raise ValueError(f"""{neigborhood} not available, choose from
            {', '.join(neighbor_funcs.keys())}""")

    TSP.neighbor = staticmethod(neighbor_func)
    TSP.state_init = staticmethod(init_func)
    return TSP

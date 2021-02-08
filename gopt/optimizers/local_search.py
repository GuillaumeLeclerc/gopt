import numpy as np

from .base import Optimizer

def LocalSearch(Problem):

    problem = Problem.compile()

    # We have to extract the functions from the namespace object
    # Otherwise numba will be confused
    loss = problem.loss
    copy_state = problem.copy_state
    neighbor_loss = problem.neighbor_loss

    class LocalSearch(Optimizer):
        # This optimizer doesn't need anything other than
        # the state of solutions
        state_dtype = np.dtype([
            ('is_first_iter', np.bool)
         ])

        # Needs two solutions, the current best one (by convention at index 0)
        # and the current one
        states_required = 2

        @staticmethod
        def init(my_state, _):
            my_state['is_first_iter'] = True

        @staticmethod
        def step(my_state, solution_states, solution_losses, problem_data,
                 iterations):

            # No need to recompute the loss if we already computed it
            # before
            if my_state['is_first_iter']:
                best_so_far = loss(solution_states[0], problem_data)
                my_state['is_first_iter'] = False
            else:
                best_so_far = solution_losses[0]

            copy_state(solution_states, 0, solution_states, 1)

            for _ in range(iterations):
                new_loss = neighbor_loss(solution_states[1], problem_data,
                                         best_so_far)
                if new_loss < best_so_far:
                    best_so_far = new_loss
                    copy_state(solution_states, 1, solution_states, 0)
                else:
                    copy_state(solution_states, 0, solution_states, 1)

            solution_losses[0] = best_so_far

            return best_so_far

    LocalSearch.Problem = Problem

    return LocalSearch

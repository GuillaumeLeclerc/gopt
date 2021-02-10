import numpy as np
import numba

from .base import Optimizer

def RandomLocalSearch(Problem):

    problem = Problem.compile()

    # We have to extract the functions from the namespace object
    # Otherwise numba will be confused
    copy_state = problem.copy_state
    neighbor_loss = problem.neighbor_loss

    problem_nh_dim = np.array(Problem.neighbor_dimensionality).astype('int32')
    print(problem_nh_dim, problem_nh_dim.shape)

    class RandomLocalSearch(Optimizer):
        # This optimizer doesn't need anything other than
        # the state of solutions
        state_dtype = None

        # Needs two solutions, the current best one (by convention at index 0)
        # and the current one
        states_required = 2

        @staticmethod
        def step(my_state, solution_states, solution_losses, problem_data,
                 iterations):

            # The original loss has been computed in the init function
            best_so_far = solution_losses[0]

            copy_state(solution_states, 0, solution_states, 1)

            direction = np.zeros(len(problem_nh_dim), dtype='int32')

            for _ in range(iterations):
                for i, max_val in enumerate(problem_nh_dim):
                    direction[i] = np.random.randint(0, max_val)
                new_loss = neighbor_loss(solution_states[1], problem_data,
                                         direction, best_so_far)
                if new_loss < best_so_far:
                    best_so_far = new_loss
                    copy_state(solution_states, 1, solution_states, 0)
                else:
                    copy_state(solution_states, 0, solution_states, 1)

            solution_losses[0] = best_so_far

            return best_so_far

    RandomLocalSearch.Problem = Problem

    return RandomLocalSearch

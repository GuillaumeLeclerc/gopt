from .base import Optimizer

class LocalSearch(Optimizer):
    # This optimizer doesn't need anything other than the state of solutions
    state_dtype = None

    # Needs two solutions, the current best one (by convention at index 0)
    # and the current one
    states_required = 2

    @staticmethod
    def generate_step_code(Problem):
        problem = Problem.compile()

        # We have to extract the functions from the namespace object
        # Otherwise numba will be confused
        loss = problem.loss
        copy_state = problem.copy_state
        neighbor_loss = problem.neighbor_loss

        def step(_, solution_states, problem_data, iterations):
            best_so_far = loss(solution_states[0], problem_data)
            copy_state(solution_states, 0, solution_states, 1)

            for _ in range(iterations):
                new_loss = neighbor_loss(solution_states[1], problem_data,
                                         best_so_far)
                if new_loss < best_so_far:
                    best_so_far = new_loss
                    copy_state(solution_states, 1, solution_states, 0)
                else:
                    copy_state(solution_states, 0, solution_states, 1)
            return best_so_far

        return step

import numpy as np
import logging

from .base import Runner
from .code_runner import CodeRunner


class CPURunner(Runner):

    def __init__(self, Shuffler, problem_data, num_cores):
        super().__init__(Shuffler, problem_data)

    def run(self, max_iter=None, max_time=None):

        opt_state_dtype = self.Optimizer.state_dtype

        def to_run(query_vector, shuffler_state, solutions, losses,
                   problem_data, iterations):

            it_left = iterations
            pop_size, num_iterations = self.shuffler_code.schedule_work(
                query_vector,
                shuffler_state[0],
                solutions,
                losses,
                it_left
            )

            best_loss = np.inf
            for pop_ix in range(pop_size):
                pop_id = query_vector[pop_ix]

                if opt_state_dtype is None:
                    opt_states = None
                else:
                    opt_states = self.optimizer_states[pop_id]

                closs = self.optimizer_code.step(
                    opt_states,
                    solutions[pop_id],
                    losses[pop_id],
                    problem_data,
                    num_iterations
                )

                best_loss = min(best_loss, closs)

            return best_loss


        code_runner = CodeRunner(max_iter=max_iter,
                                 max_time=max_time)


        while True:
            try:
                code_runner.run_block(to_run,
                                      self.query_vector,
                                      self.optimizer_states,
                                      self.solution_states,
                                      self.solution_losses,
                                      self.problem_data)

            except (KeyboardInterrupt, StopIteration):
                break

        final_result_ix = self.shuffler_code.final_result(
            self.shuffler_state,
            self.solution_states,
            self.solution_losses,
            self.Shuffler.population_size
        )

        return self.solution_losses[final_result_ix, 0], self.solution_states[final_result_ix, 0]




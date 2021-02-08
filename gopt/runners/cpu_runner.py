from types import SimpleNamespace
import logging
import numpy as np
import numba

from .base import Runner
from .code_runner import CodeRunner
from ..compiler import Compilable, Compiler


class CPURunner(Runner):

    def __init__(self, Shuffler, problem_data, num_cores):
        super().__init__(Shuffler, problem_data)
        self.num_cores = num_cores

    def run(self, max_iter=None, max_time=None):
        previous_numba_core_count = numba.get_num_threads()
        numba.set_num_threads(self.num_cores)

        runner_code = self.compile()

        code_runner = CodeRunner(max_iter=max_iter,
                                 max_time=max_time)

        while True:
            try:
                code_runner.run_block(runner_code.to_run,
                                      self.query_vector,
                                      self.shuffler_state,
                                      self.solution_states,
                                      self.solution_losses,
                                      self.problem_data,
                                      self.optimizer_states)

            except (KeyboardInterrupt, StopIteration):
                break

        final_result_ix = self.shuffler_code.final_result(
            self.shuffler_state,
            self.solution_states,
            self.solution_losses,
            self.Shuffler.population_size
        )

        numba.set_num_threads(previous_numba_core_count)

        return (self.solution_losses[final_result_ix, 0],
                self.solution_states[final_result_ix, 0])


    def is_compiled(self):
        return self._compiled is not None

    def compile(self):
        if self.is_compiled():
            return self._compiled

        opt_state_dtype = self.Optimizer.state_dtype

        schedule_work = self.shuffler_code.schedule_work
        step = self.optimizer_code.step

        def to_run(query_vector, shuffler_state, solutions, losses,
                   problem_data, optimizer_states, iterations):

            it_left = iterations
            while it_left > 0:
                pop_size, num_iterations = schedule_work(
                    query_vector,
                    shuffler_state,
                    solutions,
                    losses,
                    it_left
                )
                num_iterations = min(num_iterations, it_left)

                best_loss = np.inf
                for pop_ix in numba.prange(pop_size):
                    pop_id = query_vector[pop_ix]

                    if opt_state_dtype is None:
                        opt_states = None
                    else:
                        opt_states = optimizer_states[pop_id]

                    closs = step(
                        opt_states,
                        solutions[pop_id],
                        losses[pop_id],
                        problem_data,
                        num_iterations
                    )

                    best_loss = min(best_loss, closs)

                it_left -= num_iterations

            return best_loss

        solution_state_ntype = numba.types.Array(
            self.Optimizer.Problem.state_ntype, 2, 'C')
        solution_losses_ntype = numba.types.Array(Compiler.loss_ntype, 2, 'C')
        query_vector_ntype = numba.types.Array(numba.int32, 1, 'C')
        if self.Optimizer.state_dtype is None:
            optimizer_states = numba.typeof(None)
        else:
            optimizer_states = numba.types.Array(self.Optimizer.state_ntype, 1, 'C')


        to_run_signature = numba.float32(
            query_vector_ntype,
            self.Shuffler.state_ntype,
            solution_state_ntype,
            solution_losses_ntype,
            self.Problem.pdata_ntype,
            optimizer_states,
            numba.int32
        )

        compiled_to_run = Compiler.jit(type(self).__name__, 'to_run',
                                       to_run_signature, to_run, parallel=True)

        self._compiled = SimpleNamespace(
            to_run=compiled_to_run
        )

        return self._compiled


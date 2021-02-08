import numpy as np
from abc import ABCMeta

from ..compiler import Compiler, Compilable
from .code_runner import CodeRunner

class Runner(metaclass=ABCMeta):

    def __init__(self, Shuffler, problem_data):
        self._compiled = None

        # super().__init__(Shuffler, problem_data)
        self.Shuffler = Shuffler
        self.Optimizer = Shuffler.Optimizer
        self.Problem = self.Optimizer.Problem
        self.problem_data = problem_data

        # Compilation
        #############
        self.problem_code = self.Problem.compile()
        self.optimizer_code = self.Optimizer.compile()
        self.shuffler_code = self.Shuffler.compile()
        self.query_vector_alloc = Compiler.generate_allocator(
            type(self).__name__, np.int32)

        self.losses_alloc = Compiler.generate_allocator(
            type(self).__name__, Compiler.loss_dtype)

        # Memory Allocation
        ###################
        self.solution_states = self.problem_code.allocator(
            self.Shuffler.population_size,
            self.Optimizer.states_required
        )
        self.optimizer_states = self.optimizer_code.allocator(
            self.Shuffler.population_size
        )

        self.shuffler_state = self.shuffler_code.allocator(1)
        if self.shuffler_state is not None:
            self.shuffler_state = self.shuffler_state[0]

        self.query_vector = self.query_vector_alloc(
            self.Shuffler.population_size)

        self.solution_losses = self.losses_alloc(
            self.Shuffler.population_size,
            self.Optimizer.states_required
        )

        self.solution_losses.fill(-1)

        # State Initialization
        ######################

        # TODO consider compiling this (prob not very useful though)
        for pop_id in range(self.Shuffler.population_size):
            # If the optimizer has no state we don't have to initialize it
            if self.optimizer_states is not None:
                self.optimizer_code.init_state(self.optimizer_states[pop_id],
                                               problem_data)  # TODO should not need this argument
            self.problem_code.init_state(self.solution_states[pop_id, 0],
                                         problem_data)

        self.shuffler_code.init(self.shuffler_state, self.query_vector)

    def run(self, max_iter=None, max_time=None):
        raise NotImplementedError

        code_runner = CodeRunner(max_iter=max_iter,
                                 max_time=max_time)
        while True:
            try:
                code_runner.run_block(self.optimizer_code.step_code,
                                      self.optimizer_states,
                                      self.solution_states,
                                      self.problem_data)

            except (KeyboardInterrupt, StopIteration):
                break

        solution = self.solution_states[0]
        return self.problem_code.loss(solution, self.problem_data), solution

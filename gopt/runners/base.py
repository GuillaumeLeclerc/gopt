import logging
from time import time
from abc import ABCMeta
import numpy as np

from ..compiler import Compiler, Compilable
from .code_runner import CodeRunner

base_logger = logging.getLogger('gopt')

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

        # State Initialization
        ######################

        self.logger = base_logger.getChild(type(self).__name__)
        self.logger.info('Start initializing states')

        start_time = time()
        # TODO consider compiling this (prob not very useful though)
        for pop_id in range(self.Shuffler.population_size):
            self.problem_code.init_state(self.solution_states[pop_id, 0],
                                         problem_data)

            # Compute the loss of the newly inited solution
            self.solution_losses[pop_id, 0] = self.problem_code.loss(
                self.solution_states[pop_id, 0],
                self.problem_data
            )

            # Even if the optimizer has state we might still init it
            # It could write to the loss vector
            if self.optimizer_states is not None:
                opt_state = self.optimizer_states[pop_id]
            else:
                opt_state = None

            self.optimizer_code.init_state(opt_state,
                                           self.solution_states[pop_id],
                                           self.solution_losses[pop_id],
                                           problem_data)

        self.shuffler_code.init(self.shuffler_state, self.query_vector)
        self.logger.info(
            f'Done initializing states({time() - start_time:.2f}sec)')



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

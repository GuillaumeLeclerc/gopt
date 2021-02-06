import logging

from .code_runner import CodeRunner

class SingleCoreCPURunner:

    def __init__(self, Problem, Optimizer, problem_data):
        self.Problem = Problem
        self.Optimizer = Optimizer
        self.problem_data = problem_data

        self.problem_code = Problem.compile()
        self.optimizer_code = Optimizer.compile(Problem)

        states_required = Optimizer.states_required
        self.solution_states = self.problem_code.allocator(states_required)

        self.optimizer_state = self.optimizer_code.allocator()

        self.problem_code.init_state(self.solution_states[0], problem_data)
        self.optimizer_code.init_state(self.optimizer_state, problem_data)

    def run(self, max_iter=None, max_time=None):
        code_runner = CodeRunner(max_iter=max_iter,
                                 max_time=max_time)
        while True:
            try:
                code_runner.run_block(self.optimizer_code.step_code,
                                      self.optimizer_state,
                                      self.solution_states,
                                      self.problem_data)

            except (KeyboardInterrupt, StopIteration):
                break

        solution = self.solution_states[0]
        return self.problem_code.loss(solution, self.problem_data), solution

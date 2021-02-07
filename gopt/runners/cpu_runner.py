import numpy as np
import logging

from .base import Runner


class CPURunner(Runner):

    def __init__(self, Shuffler, problem_data, num_cores):
        super().__init__(Shuffler, problem_data)

    def run(self, max_iter=None, max_time=None):
        raise NotImplementedError


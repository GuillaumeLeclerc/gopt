import numba
from types import SimpleNamespace
import numpy as np

from ..compiler import Compiler

#
# The goal of this class is to describe a strategy used locally to accept
# or reject a new neiboring solution
class Optimizer:
    ######
    # Should be provided by the user
    ######

    # This is the type of state kept by an optimizer
    state_dtype = None

    # This describe how many problem states needs to be allocated for this
    # optimizer. It is usually at least 2, one for the current state and one
    # for the best so far but it can be as much as needed.
    states_required = 2

    # This is the function that will be run to improve the current best
    # solution
    # Since this is compiled and really fast we need to run a decent
    # number of iterations before going back to python
    #
    # Please note the lack of *self* as the first argument!
    @staticmethod
    def generate_step_code(Problem):
        def step(my_state, solution_states, problem_data, iterations):
            raise NotImplementedError
        raise NotImplementedError

    # Since the generated code depends on the Problem, it has to be
    # recompiled all the time
    @classmethod
    def compile(cls, Problem):
        code = cls.generate_step_code(Problem)

        if cls.state_dtype is None:
            cls.state_ntype = numba.void
        else:
            cls.state_ntype = numba.typeof(cls.state_dtype).dtype

        # Should be the signature of the function returned by
        # generate_state_code
        signature = numba.float32(  # Return the loss of its best solution
            cls.state_ntype,
            numba.types.Array(Problem.state_ntype, 1, 'C'),
            Problem.pdata_ntype,
            numba.int64
        )

        if cls.state_dtype is None:
            def allocator(size=1):
                return None
        else:
            def allocator(size=1):
                return np.empty(shape=size, dtype=cls.state_dtype)

        compiled = Compiler.jit('optimizer', signature, code)

        return SimpleNamespace(
            allocator=allocator,
            step_code=compiled,
        )

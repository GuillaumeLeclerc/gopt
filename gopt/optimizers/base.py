import numba
from types import SimpleNamespace
import numpy as np
from abc import ABC, abstractmethod

from ..compiler import Compiler

#
# The goal of this class is to describe a strategy used locally to accept
# or reject a new neiboring solution
class Optimizer(ABC):
    ######
    # Should be replaced by the user
    ######
    #
    # This is the type of state kept by an optimizer
    state_dtype = None
    #
    # This describe how many problem states needs to be allocated for this
    # optimizer. It is usually at least 2, one for the current state and one
    # for the best so far but it can be as much as needed.
    states_required = 2
    #############

    # This is the function that will be run to improve the current best
    # solution
    # Since this is compiled and really fast we need to run a decent
    # number of iterations before going back to python
    #
    # Please note the lack of *self* as the first argument!
    @staticmethod
    @abstractmethod
    def generate_step_code(Problem):
        def step(my_state, solution_states, problem_data, iterations):
            raise NotImplementedError
        raise NotImplementedError

    # This is the function called to initialize the state of the optimizer
    @classmethod
    def generate_init_code(cls, Problem):
        if cls.state_dtype is not None:
            m = "generate_step_code has to be impleted on stateful optimizers"
            raise NotImplementedError(m)

        def init(my_state, problem_data):
            pass

        return init

    # Since the generated code depends on the Problem, it has to be
    # recompiled all the time
    @classmethod
    def compile(cls, Problem):
        if cls.state_dtype is None:
            cls.state_ntype = numba.void
        else:
            cls.state_ntype = numba.typeof(cls.state_dtype).dtype

        # Should be the signature of the function returned by
        # generate_state_code
        step_signature = numba.float32(  # Return the loss of its best solution
            cls.state_ntype,
            numba.types.Array(Problem.state_ntype, 1, 'C'),
            Problem.pdata_ntype,
            numba.int64
        )

        # Should be the signature of the function returned by
        # generate_init_code
        init_signature = numba.void(
            cls.state_ntype,
            Problem.pdata_ntype,
        )

        if cls.state_dtype is None:
            def allocator(size=1):
                return None
        else:
            def allocator(size=1):
                return np.empty(shape=size, dtype=cls.state_dtype)

        step_code = cls.generate_step_code(Problem)
        compiled_step = Compiler.jit(cls.__name__, 'step function',
                                     step_signature, step_code)

        init_code = cls.generate_init_code(Problem)
        compiled_init = Compiler.jit(cls.__name__, 'optimizer init function',
                                     init_signature, init_code)

        return SimpleNamespace(
            allocator=allocator,
            step_code=compiled_step,
            init_state=compiled_init
        )

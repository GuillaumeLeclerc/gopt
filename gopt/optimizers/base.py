import numba
from types import SimpleNamespace
import numpy as np
from abc import ABCMeta, abstractmethod

from ..compiler import Compiler, Compilable

#
# The goal of this class is to describe a strategy used locally to accept
# or reject a new neiboring solution
class Optimizer(Compilable, metaclass=ABCMeta):
    ######
    # Should be replaced by the user
    ######
    #
    # This is the type of state kept by an optimizer
    state_dtype = None
    # This is the problem solved by this optimizer
    # We need to know it in advance so that compilation depend on it
    Problem = None
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
    def step(my_state, solution_states, solution_losses, problem_data,
             iterations):
        raise NotImplementedError

    # This is the function called to initialize the state of the optimizer
    #
    # Please note the lack of *self* as the first argument!
    @staticmethod
    def init(my_state, problem_data):
        pass

    # Since the generated code depends on the Problem, it has to be
    # recompiled all the time
    @classmethod
    def compile(cls):
        if cls.is_compiled():
            return cls._compiled

        if cls.Problem is None:
            raise AssertionError("Problem attribute needs to be provided")

        if cls.state_dtype is None:
            cls.state_ntype = numba.void
        else:
            cls.state_ntype = numba.typeof(cls.state_dtype).dtype

        # Should be the signature of the function returned by
        # generate_state_code
        step_signature = numba.float32(  # Return the loss of its best solution
            cls.state_ntype,
            numba.types.Array(cls.Problem.state_ntype, 1, 'C'),
            Compiler.loss_array_ntype,
            cls.Problem.pdata_ntype,
            numba.int64
        )

        # Should be the signature of the function returned by
        # generate_init_code
        init_signature = numba.void(
            cls.state_ntype,
            cls.Problem.pdata_ntype,
        )

        state_allocator = Compiler.generate_allocator(cls.__name__,
                                                      cls.state_dtype)

        compiled_step = Compiler.jit(cls.__name__, 'step function',
                                     step_signature, cls.step)

        compiled_init = Compiler.jit(cls.__name__, 'optimizer init function',
                                     init_signature, cls.init)

        cls._compiled = SimpleNamespace(
            allocator=state_allocator,
            step=compiled_step,
            init_state=compiled_init
        )

        return cls._compiled

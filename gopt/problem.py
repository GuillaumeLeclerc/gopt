import numba
import logging
import numpy as np
from types import SimpleNamespace

from .compiler import Compiler, Compilable

class Problem(Compilable):
    ######
    # Should be provided by the user
    ######

    problem_name = 'Generic problem'  # Name of your problem for logging
    state_dtype = None  # The data type of solution to the problem
    problem_data_dtype = None  # The dtype of the data describing an instance

    # Fill a state with an initial solution
    #
    # Please note the lack of *self* as the first argument!
    @staticmethod
    def state_init(state, problem_data):
        raise NotImplementedError

    # The loss to minimize
    #
    # Please note the lack of *self* as the first argument!
    @staticmethod
    def loss(state_array, problem_data):
        raise NotImplementedError

    # Moves to a potential neighboring solution
    # It can optionnally return a floating point that will represent the
    # loss improvement. If this value is provided the strategy will not
    # recompute the loss
    #
    # Please note the lack of *self* as the first argument!
    @staticmethod
    def neighbor(state, problem_data):
        raise NotImplementedError

    ######
    # GOPT internals, do not overwrite!
    ######

    @classmethod
    def compile(cls):
        if cls.is_compiled():
            logging.debug(f'Problem {cls.problem_name} already compiled')
            return cls._compiled

        logging.info(f'Compiling problem {cls.problem_name}')

        # Types
        state_ntype = numba.typeof(cls.state_dtype).dtype
        pdata_ntype = numba.typeof(cls.problem_data_dtype).dtype

        # Signatures
        loss_signature = numba.float32(state_ntype, pdata_ntype)
        neighbor_signature = numba.optional(numba.float32)(state_ntype,
                                                           pdata_ntype)
        copy_signature = numba.void(numba.types.Array(state_ntype, 1, 'C'),
                                    numba.int64,
                                    numba.types.Array(state_ntype, 1, 'C'),
                                    numba.int64)
        init_state_signature = numba.void(state_ntype, pdata_ntype)

        cls.state_ntype = state_ntype
        cls.pdata_ntype = pdata_ntype

        # Generic implementations
        def copy(source, source_ix, dest, dest_ix):
            dest[dest_ix:dest_ix + 1] = source[source_ix:source_ix + 1]

        def allocator(size=1):
            return np.empty(shape=size, dtype=cls.state_dtype)

        # Compilation of all the functions
        copy = Compiler.jit('copier', copy_signature, copy)
        loss = Compiler.jit('loss', loss_signature, cls.loss)
        init = Compiler.jit('initializer', init_state_signature,
                            cls.state_init)
        neighbor = Compiler.jit('movement function', neighbor_signature,
                                cls.neighbor)

        cls._compiled = SimpleNamespace(allocator=allocator, copy_state=copy,
                                        init_state=init, loss=loss,
                                        neighbor=neighbor)

        return cls._compiled

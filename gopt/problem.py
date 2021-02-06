import numba
import logging
import numpy as np
from types import SimpleNamespace
from abc import ABCMeta, abstractmethod

from .compiler import Compiler, Compilable

class Problem(Compilable, metaclass=ABCMeta):
    ######
    # Should be provided by the user
    ######
    #
    problem_name = 'Generic problem'  # Name of your problem for logging
    state_dtype = None  # The data type of solution to the problem
    problem_data_dtype = None  # The dtype of the data describing an instance
    ########################

    # Fill a state with an initial solution
    #
    # Please note the lack of *self* as the first argument!
    @staticmethod
    @abstractmethod
    def state_init(state, problem_data):
        raise NotImplementedError

    # The loss to minimize
    #
    # Please note the lack of *self* as the first argument!
    @staticmethod
    @abstractmethod
    def loss(state_array, problem_data):
        raise NotImplementedError

    # Moves to a potential neighboring solution
    # It can optionnally return a floating point that will represent the
    # loss improvement. If this value is provided the strategy will not
    # recompute the loss
    #
    # Please note the lack of *self* as the first argument!
    @staticmethod
    @abstractmethod
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

        # Types
        state_ntype = numba.typeof(cls.state_dtype).dtype
        pdata_ntype = numba.typeof(cls.problem_data_dtype).dtype

        # Signatures
        loss_signature = numba.float32(state_ntype, pdata_ntype)
        neighbor_signature = numba.optional(numba.float32)(state_ntype,
                                                           pdata_ntype)
        neighbor_loss_signature = numba.float32(state_ntype,
                                                pdata_ntype,
                                                numba.optional(numba.float32))
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
        copy = Compiler.jit(cls.__name__, 'copier', copy_signature, copy)
        loss = Compiler.jit(cls.__name__, 'loss', loss_signature, cls.loss)
        init = Compiler.jit(cls.__name__, 'initializer', init_state_signature,
                            cls.state_init)
        neighbor = Compiler.jit(cls.__name__, 'movement function', neighbor_signature,
                                cls.neighbor)

        def pre_neighbor_loss(state, problem_data, previous_loss=None):
            loss_delta = neighbor(state, problem_data)

            if previous_loss is None or loss_delta is None:
                return loss(state, problem_data)
            else:
                return previous_loss + loss_delta

        if Compiler.debug:
            def neighbor_loss(state, problem_data, previous_loss=None):
                result = pre_neighbor_loss(state, problem_data, previous_loss)
                reference = loss(state, problem_data)
                if not np.isclose(result, reference):
                    m = ("Loss delta from neighbor function incorrect"
                         + "(Got " + str(result) + ", expected "
                         + str(reference) + ")")
                    raise AssertionError(m)
                return result
        else:
            neighbor_loss = pre_neighbor_loss

        neighbor_loss = Compiler.jit(cls.__name__, 'neighbor+loss combined function',
                                     neighbor_loss_signature,
                                     neighbor_loss)
        cls._compiled = SimpleNamespace(allocator=allocator, copy_state=copy,
                                        init_state=init, loss=loss,
                                        neighbor=neighbor,
                                        neighbor_loss=neighbor_loss)

        return cls._compiled

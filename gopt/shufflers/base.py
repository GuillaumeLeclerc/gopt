import numba
from abc import ABCMeta, abstractmethod
from types import SimpleNamespace

from ..compiler import Compiler, Compilable


class Shuffler(Compilable, metaclass=ABCMeta):

    ######
    # Should be replaced by the user
    ######
    #
    # This is the type of state kept by the shuffler
    # Can be kept as None if no state is needed
    state_dtype = None
    # This is a reference the optimizer this Shuffler is running
    Optimizer = None
    # One should set the actual population size
    population_size = None
    ######

    # This function should return a tuple:
    #  - int, describing which individuals of the population should be run
    #  - int, saying how many iterations shoulr be ran before the next shuffle
    #
    #  The indices of the population that should be run should be written
    #  in the query_vector array
    #
    # Please note the lack of *self* as the first argument!
    @staticmethod
    @abstractmethod
    def schedule_work(query_vector, shuffler_state, solution_states,
                      solution_losses, total_iterations):
        raise NotImplementedError

    # This function implements the shuffle operation
    #
    # Please note the lack of *self* as the first argument!
    @staticmethod
    @abstractmethod
    def shuffle(shuffler_state, solution_states,
                solution_losses):
        raise NotImplementedError

    # This is the function called to initialize the state of the shuffler
    #
    # This only has to be overriden if the state_dtype is not None or if one
    # want to init the query_vector once and for all
    #
    # Please note the lack of *self* as the first argument!
    @staticmethod
    def init(my_state, query_vector):
        pass

    # This is the function to return the final solution and its loss
    #
    # Population size could reasonably be compiled and not be an argument
    # however it has a couple of drawbacks:
    # - We could generate the code in compile() based on the provided
    #   population_size but that would mean users can't override it
    # - At the moment this function is defined we don't know population size
    #   so we would have to have all implementations override this function.
    #   But since we expect most Shuffler to have the same one it's too
    #   much added complication
    #
    # Please note the lack of *self* as the first argument!
    @staticmethod
    def final_result(shuffler_state, solution_states, solution_losses,
                     population_size):
        best_ix = 0
        best_loss = solution_losses[0][0]
        for i in range(1, population_size):
            if solution_losses[i][0] < best_loss:
                best_loss = solution_losses[i][0]
                best_ix = i

        return best_ix

    # Compile the current shuffler
    @classmethod
    def compile(cls):
        if cls.is_compiled():
            return cls._compiled

        if cls.state_dtype is None:
            cls.state_ntype = numba.void
        else:
            cls.state_ntype = numba.typeof(cls.state_dtype).dtype

        solution_state_ntype = numba.types.Array(
            cls.Optimizer.Problem.state_ntype, 2, 'C')
        solution_losses_ntype = numba.types.Array(Compiler.loss_ntype, 2, 'C')
        query_vector_ntype = numba.types.Array(numba.int32, 1, 'C')

        # Shoud match the signature of cls.schedule_work(...)
        schedule_work_ret_type = numba.types.Tuple((numba.int32, numba.int32))
        schedule_work_signature = schedule_work_ret_type(
            query_vector_ntype,
            cls.state_ntype,
            solution_state_ntype,
            solution_losses_ntype,
            numba.int32
        )

        # Should match the signature of cls.shuffle(...)
        shuffle_signature = numba.void(
            cls.state_ntype,
            solution_state_ntype,
            solution_losses_ntype
        )

        init_signature = numba.void(
            cls.state_ntype,
            query_vector_ntype
        )

        final_result_signature = numba.int32(
            cls.state_ntype,
            solution_state_ntype,
            solution_losses_ntype,
            numba.int32
        )

        allocator = Compiler.generate_allocator(cls.__name__, cls.state_dtype)

        compiled_schedule_work = Compiler.jit(cls.__name__, 'schedule_work',
                                              schedule_work_signature,
                                              cls.schedule_work)

        compiled_shuffle = Compiler.jit(cls.__name__, 'shuffle',
                                        shuffle_signature,
                                        cls.shuffle)

        compiled_init = Compiler.jit(cls.__name__, 'init',
                                     init_signature,
                                     cls.init)

        compiled_final_result = Compiler.jit(cls.__name__, 'final_result',
                                             final_result_signature,
                                             cls.final_result)

        cls._compiled = SimpleNamespace(
            schedule_work=compiled_schedule_work,
            shuffle=compiled_shuffle,
            allocator=allocator,
            init=compiled_init,
            final_result=compiled_final_result
        )

        return cls._compiled

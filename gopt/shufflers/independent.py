from .base import Shuffler

def IndependentShuffler(Optimizer, population_size):

    Optimizer.compile()

    class IndependentShuffler(Shuffler):

        @staticmethod
        def schedule_work(query_vector, shuffler_state, solution_states,
                          solution_losses, total_iterations):
            return population_size, total_iterations

        @staticmethod
        def init(shuffler_state, query_vector):
            for i in range(population_size):
                query_vector[i] = i

        # Optimizers are independent in the IndependentShuffler
        @staticmethod
        def shuffle(shuffler_state, solution_states,
                    solution_losses):
            pass

    IndependentShuffler.Optimizer = Optimizer
    IndependentShuffler.population_size = population_size

    return IndependentShuffler


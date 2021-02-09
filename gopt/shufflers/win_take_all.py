from .base import Shuffler

def WinnerTakesAll(Optimizer, population_size):

    Optimizer.compile()

    class WinnerTakesAll(Shuffler):

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
            best_ix = 0
            best_loss = solution_losses[0][0]
            for i in range(1, population_size):
                if solution_losses[i][0] < best_loss:
                    best_loss = solution_losses[i][0]
                    best_ix = i

            for i in range(population_size):
                solution_states[i][0] = solution_states[best_ix][0]
                solution_losses[i][0] = solution_losses[best_ix][0]

    WinnerTakesAll.Optimizer = Optimizer
    WinnerTakesAll.population_size = population_size

    return WinnerTakesAll


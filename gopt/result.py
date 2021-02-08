class Result:

    def __init__(self, loss, solution, Problem):
        self.loss = loss
        self.solution = solution
        self.Problem = Problem
        self.runtime_info = {}


    def set_runtime_info(self, **kwargs):
        for k, v in kwargs.items():
            self.runtime_info[k] = v

    def __repr__(self):
        return f"Result<{self.Problem.__name__}>(loss={self.loss})"

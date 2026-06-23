import random

from matharena.solvers import StaticMathAgent
from matharena.solvers.math_core import Merger, Selector, Solver
class RSAAgent(StaticMathAgent):
    def __init__(self, batch_idx, problem_idx, run_idx, solver_config, default_prompt_template,
                 default_api_client_args):
        super().__init__(batch_idx, problem_idx, run_idx, solver_config, default_prompt_template,
                         default_api_client_args)
        self.log_index = f"RSAAgent-P{self.problem_idx}-R{self.run_idx}"

    def sample_blocks(self, items, block_size, seed=None):
        count = len(items)
        if count == 0:
            return []

        rng = random.Random(seed)
        perm = items[:]
        rng.shuffle(perm)

        offsets = rng.sample(range(count), min(block_size, count))
        return [[perm[(start + offset) % count] for offset in offsets] for start in range(count)]

    def create_agent(self):
        solutions = [Solver() for _ in range(self.scaffold_config["n_solutions"])]

        for i in range(self.scaffold_config["n_rounds"]):
            new_solutions = []
            blocks = self.sample_blocks(solutions, self.scaffold_config["block_size"], seed=i)
            for block in blocks:
                merged_solution = Merger(block)
                new_solutions.append(merged_solution)
            solutions = new_solutions[:]
        
        if len(solutions) == 1:
            return solutions[0]

        if self.scaffold_config["select_pairwise"]:
            while len(solutions) > 1:
                new_solutions = []
                for i in range(0, len(solutions), 2):
                    if i + 1 < len(solutions):
                        pair = [solutions[i], solutions[i + 1]]
                        selected_solution = Selector(pair)
                    else:
                        selected_solution = solutions[i]
                    new_solutions.append(selected_solution)
                solutions = new_solutions[:]
                random.shuffle(solutions)
            return solutions[0]

        return Selector(solutions)

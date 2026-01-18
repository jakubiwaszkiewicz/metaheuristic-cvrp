from loader import single_instance_loader
from constraints import Constraints
from project.solvers.base import CVRPSolver
from project.solvers.greedy_solver import GreedyCVRPSolver
from project.solvers.random_solver import RandomCVRPSolver
from project.solvers.GA_solver import GACVRPSolver
from project.solvers.SA_solver import SACVRPSolver
from statistics import mean, stdev
from project.custom_logger import print_results_table
from tqdm import tqdm
from dataclasses import dataclass
from multiprocessing import Pool

@dataclass
class SolverRun:
    name: str
    solver: 'CVRPSolver'
    no_of_runs: int

# --- Default parameters ---
GA_DEFAULTS = dict(
    pop_size=300,
    generations=1000,
    Px=0.9,
    Pm=0.02,
    tour_size=25,
    population_graph = False,
    mutation_type="inversion",
    selection_type="tournament"
)

SA_DEFAULTS = dict(
    initial_temp=1000.0,
    final_temp=0.1,
    alpha=0.99,
    max_iter_per_temp=500,
    cooling_step_graph=False,
)

# --- Parameter variations ---
GA_VARIATIONS = {
#    "pop_size": [50, 300, 1000, 10000],
#    "generations": [50, 200, 500, 1000, 10000],
#    "Px": [0.5, 0.9, 0.95, 0.999],
#    "Pm": [0.05, 0.1, 0.2, 0.5],
#    "tour_size": [3, 6, 10, 25],
#    "mutation_type": ["inversion"],
#    "selection_type": ["roulette"],
}

SA_VARIATIONS = {
#    "initial_temp": [500.0, 800.0, 1500.0, 2000.0, 3000.0],
#    "final_temp": [0.01, 0.5, 2.0, 10.0],
#    "alpha": [0.9, 0.95, 0.995],
#    "max_iter_per_temp": [50, 300, 1000],
}


def generate_solver_runs():
    solvers_runs = [
        #SolverRun(name="RANDOM_x10000", solver=RandomCVRPSolver(), no_of_runs=10000),
        #SolverRun(name="GREEDY_xN", solver=GreedyCVRPSolver(), no_of_runs=1),
        SolverRun(name="GA_default_x10", solver=GACVRPSolver(**GA_DEFAULTS), no_of_runs=10),
        #SolverRun(name="SA_default_x10", solver=SACVRPSolver(**SA_DEFAULTS), no_of_runs=10),
    ]

    # GA Variations

    if len(GA_VARIATIONS) != 0 or GA_VARIATIONS is not None:
        for param, values in GA_VARIATIONS.items():
            for val in values:
                params = GA_DEFAULTS.copy()
                params[param] = val
                name = f"GA_{param}_{val}_x10"
                solvers_runs.append(SolverRun(name=name, solver=GACVRPSolver(**params), no_of_runs=10))


    # SA Variations
    if len(SA_VARIATIONS) != 0 or GA_VARIATIONS is not None:
        for param, values in SA_VARIATIONS.items():
            for val in values:
                params = SA_DEFAULTS.copy()
                params[param] = val
                name = f"SA_{param}_{val}_x10"
                solvers_runs.append(SolverRun(name=name, solver=SACVRPSolver(**params), no_of_runs=10))

    return solvers_runs


def main():
    solvers_runs = generate_solver_runs()

    instances = [
        # "./data/toy/toy.vrp",
        "./data/A-n37-k6/A-n37-k6.vrp",
        "./data/A-n39-k5/A-n39-k5.vrp",
        "./data/A-n45-k6/A-n45-k6.vrp",
        "./data/A-n48-k7/A-n48-k7.vrp",
        "./data/A-n54-k7/A-n54-k7.vrp",
        "./data/A-n60-k9/A-n60-k9.vrp",
    ]

    test_parallel(instances, solvers_runs, n_processes=6)


def run_solver_on_instance(args):
    instance_path, solver_run = args
    instance = single_instance_loader(instance_path)
    constraints = Constraints(truck_capacity=instance.truck_capacity)
    results = []

    for _ in range(solver_run.no_of_runs):
        solution = solver_run.solver.solve(instance, constraints)

        # greedy returns list
        if isinstance(solution, list):
            costs = [s.total_cost for s in solution]
            results.extend(costs)
        else:
            results.append(solution.total_cost)

    best = min(results)
    worst = max(results)
    avg = mean(results)
    std = stdev(results) if len(results) > 1 else 0.0
    return (solver_run.name, [best, worst, avg, std], instance.instance_name)

def test_parallel(instances: list[str], solvers_runs: list[SolverRun], n_processes=7):
    args_list = [(instance_path, solver_run) for instance_path in instances for solver_run in solvers_runs]
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(pool.imap(run_solver_on_instance, args_list), total=len(args_list)))

    # grupowanie wyników per instance
    instances_results = {}
    for solver_name, stats, instance_name in results:
        if instance_name not in instances_results:
            instances_results[instance_name] = {'instance': instance_name}
        instances_results[instance_name][solver_name] = stats

    print_results_table(list(instances_results.values()), solvers_runs)

def single_processor_test(instances: list[str], solvers_runs: list[SolverRun]):
    instances_results = []
    for instance_path in tqdm(instances, desc="Instances"):
        instance = single_instance_loader(instance_path)
        constraints = Constraints(truck_capacity=instance.truck_capacity)

        result_dict = {'instance': instance.instance_name}
        for solver_run in tqdm(solvers_runs, desc="Solvers"):
            results = []

            for _ in tqdm(range(solver_run.no_of_runs), desc="Runs"):
                solution = solver_run.solver.solve(instance, constraints)
                results.append(solution.total_cost)

            best = min(results)
            worst = max(results)
            avg = mean(results)
            std = stdev(results) if len(results) > 1 else 0.0
            result_dict[solver_run.name] = [best, worst, avg, std]

        instances_results.append(result_dict)
    print_results_table(instances_results, solvers_runs)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()

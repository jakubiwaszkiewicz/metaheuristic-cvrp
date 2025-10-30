import csv
import re
from tabulate import tabulate
from datetime import datetime

def parse_param_info(solver_name: str):
    """
    Wyciąga nazwę i wartość parametru z nazwy solvera, np.:
    'GA_pop_size_100_x10' -> ('pop_size', '100')
    'SA_alpha_0.9_x10' -> ('alpha', '0.9')
    Jeśli to solver bazowy (np. 'GA_default_x10'), zwraca ('default', '-')
    """
    match = re.match(r"(GA|SA)_([a-zA-Z_]+)_([a-zA-Z0-9.\-]+)_x\d+", solver_name)
    if match:
        return match.group(2), match.group(3)
    elif "default" in solver_name.lower():
        return "default", "-"
    elif "GREEDY" in solver_name:
        return "GREEDY", "-"
    elif "RANDOM" in solver_name:
        return "RANDOM", "-"
    else:
        return "unknown", "-"


def print_results_table(instances_results, solvers_runs, save_csv_path=None):
    if save_csv_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_csv_path = f"results_{timestamp}.csv"

    headers = ["Instance", "Solver", "Param Name", "Param Value", "Best", "Worst", "Avg", "Std"]
    table_data = []

    for res in instances_results:
        for solver_run in solvers_runs:
            if solver_run.name not in res:
                continue

            param_name, param_value = parse_param_info(solver_run.name)

            row = [
                res['instance'],          # Instance
                solver_run.name,          # Solver name
                param_name,               # Param name
                param_value,              # Param value
                res[solver_run.name][0],  # Best
                res[solver_run.name][1],  # Worst
                res[solver_run.name][2],  # Avg
                res[solver_run.name][3],  # Std
            ]
            table_data.append(row)

    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".3f"))

    with open(save_csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerows(table_data)

    print(f"zapisane do pliku: {save_csv_path}")
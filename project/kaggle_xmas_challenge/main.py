import math
from pathlib import Path
import yaml
import numpy as np
import argparse
import datetime
import concurrent.futures
import copy
import random
import pandas as pd
from tqdm import tqdm
from shapely.geometry import Point
from shapely.ops import unary_union
from decimal import Decimal, getcontext
from utils.data_repr import ChristmasTree
from utils.constants import scale_factor, decimal_precision
import time
from collections import defaultdict
import math

FUNC_STATS = defaultdict(lambda: {"time": 0.0, "calls": 0})

def profile(func):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        name = func.__qualname__
        FUNC_STATS[name]["time"] += dt
        FUNC_STATS[name]["calls"] += 1
        return result
    return wrapper


getcontext().prec = decimal_precision

@profile
def format_time(elapsed):
    """Take a time in seconds and return a string hh:mm:ss."""
    elapsed_rounded = int(round(elapsed))

    hours = elapsed_rounded // 3600
    minutes = (elapsed_rounded % 3600) // 60
    seconds = elapsed_rounded % 60

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

@profile
def load_configuration_from_df(n, existing_df):
    """
    Load existing configuration from submission CSV.
    """
    group_data = existing_df[existing_df["id"].str.startswith(f"{n:03d}_")]
    trees = []
    for _, row in group_data.iterrows():
        x = row["x"][1:]
        y = row["y"][1:]
        deg = row["deg"][1:]
        trees.append(ChristmasTree(x, y, deg))
    if len(trees) != n:
        raise RuntimeError("Number of trees is inconsistent")
    return trees

def to_str(x: Decimal):
    return f"s{float(x)}"

@profile
def clone_trees(trees: list[ChristmasTree]):
    cloned_trees = [tree.clone() for tree in trees]
    return cloned_trees

class SimulatedAnnealing:
    def __init__(
            self,
            initial_trees,
            a,
            b,
            num_cell,
            append_x,
            append_y,
            Tmax,
            Tmin,
            nsteps,
            iterations_per_temperature,
            cooling,
            alpha,
            position_delta,
            angle_delta,
            angle_delta2,
            delta_t,
            random_state,
            log_freq,
            F_pos=0.7,
            F_ang=0.2,
            F_lat=0.5,
            F_rot=0.15,
            temp_exponent=0.7,
    ):
        self.initial_trees = initial_trees
        self.a = a
        self.b = b
        self.num_cell = num_cell
        self.append_x = append_x
        self.append_y = append_y
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.nsteps = nsteps
        self.iterations_per_temperature = iterations_per_temperature
        self.cooling = cooling
        self.alpha = alpha
        self.position_delta = position_delta
        self.angle_delta = angle_delta
        self.angle_delta2 = angle_delta2
        self.delta_t = delta_t
        self.log_freq = log_freq
        random.seed(random_state)

        # DE parametry
        self.F_pos = Decimal(str(F_pos))
        self.F_ang = Decimal(str(F_ang))
        self.F_lat = Decimal(str(F_lat))
        self.F_rot = Decimal(str(F_rot))
        self.temp_exponent = temp_exponent

    @profile
    def _temperature_scale(self, temperature):
        return Decimal(str((temperature / self.Tmax) ** self.temp_exponent))

    @profile
    def get_lengths(self, current_trees):
        xys = np.concatenate([np.asarray(t.polygon.exterior.xy).T / 1e15 for t in current_trees])
        min_x, min_y = xys.min(axis=0)
        max_x, max_y = xys.max(axis=0)
        return max_x - min_x, max_y - min_y

    @profile
    def calculate_score(self, trees):
        a, b = self.get_lengths(trees)
        longer_side = max(a,b)
        score = longer_side ** 2 / len(trees)
        return score

    @profile
    def has_overlap(self, trees, n=None):
        """Check for overlap between trees."""
        if len(trees) <= 1:
            return False
        if n is None:
            # check all trees
            for i, tree1 in enumerate(trees):
                for j, tree2 in enumerate(trees):
                    if i < j:
                        if tree1.polygon.intersects(tree2.polygon) and not tree1.polygon.touches(tree2.polygon):
                            return True
        else:
            # check overlap of specific tree
            for i, tree1 in enumerate(trees):
                if i != n:
                    if tree1.polygon.intersects(trees[n].polygon) and not tree1.polygon.touches(trees[n].polygon):
                        return True
        return False

    @profile
    def _acceptance_probability(self, current_energy, new_energy, temperature):
        """Calculate the probability of accepting a new solution."""
        if new_energy < current_energy:
            return 1.0
        return math.exp((current_energy - new_energy) / temperature)

    @profile
    def perturb_tree(self, tree, Temperature):
        """Perturb tree position and angle"""
        old_x, old_y, old_angle = tree.get_params()

        random_change_x = Decimal(str(random.uniform(-self.position_delta, self.position_delta)))
        random_change_y = Decimal(str(random.uniform(-self.position_delta, self.position_delta)))
        random_change_angle = Decimal(str(random.uniform(-self.angle_delta, self.angle_delta)))

        scale = Decimal(str(Temperature / self.Tmax))

        random_change_x *= scale
        random_change_y *= scale
        random_change_angle *= scale

        new_x = old_x + random_change_x
        new_y = old_y + random_change_y
        new_angle = (old_angle + random_change_angle) % 360

        tree.set_params(new_x, new_y, new_angle)
        return old_x, old_y, old_angle

    @profile
    def perturb_translations(self, a, b, temperature):
        """Perturb tree position and angle"""
        old_a = Decimal(str(copy.copy(a)))
        old_b = Decimal(str(copy.copy(b)))

        da = Decimal(str(random.uniform(-self.delta_t, self.delta_t)))
        db = Decimal(str(random.uniform(-self.delta_t, self.delta_t)))

        scale = Decimal(str(temperature / self.Tmax))

        da *= scale
        db *= scale

        new_a = old_a + old_a * da
        new_b = old_b + old_b * db
        return new_a, new_b, old_a, old_b

    @profile
    def rotate_all(self, trees):
        """Perturb trees angle"""
        old_angles = []
        dangle = Decimal(str(random.uniform(-self.angle_delta2, self.angle_delta2)))
        for tree in trees:
            x, y, old_angle = tree.get_params()
            old_angles.append(old_angle)
            new_angle = (old_angle + dangle) % 360
            tree.set_params(x, y, new_angle)
        return trees, old_angles

    @profile
    def translate(self, primitive_trees, a, b, num_cell):
        lattice_trees = []
        cells_width = num_cell[0]
        cells_height = num_cell[1]
        for tree in primitive_trees:
            for width_cell in range(cells_width):
                for height_cell in range(cells_height):
                    lattice_trees.append(
                        ChristmasTree(
                            center_x=tree.center_x + Decimal(width_cell * a),
                            center_y=tree.center_y + Decimal(height_cell * b),
                            angle=tree.angle,
                        )
                    )
        return lattice_trees

    @profile
    def custom_log(self, temperature, current_score, best_score, no_of_rejections):
        rejection_pct = 100 * no_of_rejections / self.iterations_per_temperature
        print(
            f"Temperature: {temperature}, "
            f"Current score: {current_score}, "
            f"Best score: {best_score:8.5f}, "
            f"Num cells: {self.num_cell}, "
            f"Rejections: {rejection_pct:.1f}%",
            flush=True,
        )

    @profile
    def solve(self):
        temperature = self.Tmax
        primitive_trees = clone_trees(self.initial_trees)

        if self.a is None:
            a, b = self.get_lengths(primitive_trees)
        else:
            a, b = copy.copy(self.a), copy.copy(self.b)

        lattice_trees = self.translate(primitive_trees, a, b, self.num_cell)

        if self.has_overlap(lattice_trees):
            raise Exception("Initial tree configuration has overlap, implementation Error...")

        current_score = self.calculate_score(lattice_trees)
        best_trees = clone_trees(lattice_trees)
        best_score = current_score

        for cooling_step in range(self.nsteps):
            all_scores = []
            no_of_rejections = 0
            for iteration_at_temperature in range(self.iterations_per_temperature):

                move_probability = random.random()
                perturb_tree_probability = 0.8
                perturb_translations_probability = 0.8
                has_overlap_in_iter = False

                if perturb_tree_probability < move_probability:
                    tree_probability = random.randint(0,1)
                    old_params = self.perturb_tree(primitive_trees[tree_probability], temperature)
                    if self.has_overlap(self.translate(primitive_trees, a, b, self.num_cell)):
                        has_overlap_in_iter = True
                        no_of_rejections += 1
                        primitive_trees[tree_probability].set_params(*old_params)
                elif perturb_translations_probability < move_probability:
                    a, b, old_a, old_b = self.perturb_translations(a, b, temperature)
                    if self.has_overlap(self.translate(primitive_trees, a, b, self.num_cell)):
                        has_overlap_in_iter = True
                        no_of_rejections += 1
                        a = old_a
                        b = old_b
                else:
                    primitive_trees, old_angles = self.rotate_all(primitive_trees)
                    if self.has_overlap(self.translate(primitive_trees, a, b, self.num_cell)):
                        has_overlap_in_iter = True
                        no_of_rejections += 1
                        for i, tree in enumerate(primitive_trees):
                            x, y, _ = tree.get_params()
                            tree.set_params(x, y, old_angles[i])

                if not has_overlap_in_iter:
                    lattice_trees = self.translate(primitive_trees, a, b, self.num_cell)
                    new_score = self.calculate_score(lattice_trees)

                    acceptance_probability = self._acceptance_probability(current_score, new_score, temperature)
                    acceptance = acceptance_probability > random.random()

                    if acceptance:
                        current_score = new_score
                        if new_score < best_score:
                            best_score = new_score
                            best_trees = clone_trees(lattice_trees)

            self.custom_log(temperature, current_score, best_score, no_of_rejections)

            if self.cooling == "linear":
                temperature_to_tmax = (self.Tmax - self.Tmin)
                temperature_step = temperature_to_tmax / self.nsteps
                temperature -= temperature_step

            all_scores.append(best_score)
            self.save_score(temperature, best_score)
        return best_trees

    @profile
    def save_score(self, temperature, score):
        dir_name = Path("all_scores")
        dir_name.mkdir(parents=True, exist_ok=True)
        save_path = dir_name / f"scores_[{self.num_cell[0]}_{self.num_cell[1]}].csv"
        with open(save_path, "a") as f:
            f.write(f"{temperature},{score}\n")

@profile
def get_tree_list_side_length(tree_list: list[ChristmasTree]) -> Decimal:
    all_polygons = [tree.polygon for tree in tree_list]
    bounds = unary_union(all_polygons).bounds
    return Decimal(max(bounds[2] - bounds[0], bounds[3] - bounds[1])) / scale_factor

@profile
def get_total_score(dict_of_side_length: dict[str, Decimal]):
    score = 0
    for key, variable in dict_of_side_length.items():
        score += variable ** 2 / Decimal(key)
    return score

def run_sa_for_cells(args):
    i, j, initial_trees, config = args
    config_local = config["params"].copy()
    config_local["num_cell"] = [i, j]
    sa = SimulatedAnnealing(initial_trees, **config_local)
    trees = sa.solve()
    return trees

def evaluate_and_store_best(tree_list):
    N = len(tree_list)
    side = get_tree_list_side_length(tree_list)
    if N not in new_trees:
        new_trees[N] = (side, tree_list)
    else:
        old_side, _ = new_trees[N]
        if side < old_side:
            new_trees[N] = (side, tree_list)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--multicore', help='Should be run in multiprocess mode? (y/n)', required=True)
    args = vars(parser.parse_args())

    multicore = True if args['multicore'].lower() == 'y' else False

    # init trees
    initial_trees = []
    for x, y, deg in [[-4.191683864412409, -4.498489528496051, 74.54421568660419],
                      [-4.92202045352307, -4.727639556649786, 254.5401905706735]]:
        initial_trees.append(ChristmasTree(x, y, deg))

    # load yaml config
    with open("config.yaml", "r") as file_obj:
        config = yaml.safe_load(file_obj)

    max_cells = 10
    cell_combinations = [
        (i, j)
        for i in range(1, max_cells + 1)
        for j in range(1, max_cells + 1)
    ]

    new_trees = {}

    args_list = [(i, j, initial_trees, config) for (i, j) in cell_combinations]

    if multicore:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for trees in tqdm(executor.map(run_sa_for_cells, args_list), total=len(cell_combinations)):
                evaluate_and_store_best(trees)
    else:
        for cell_combination in tqdm(cell_combinations, total=len(cell_combinations)):
            i, j = cell_combination
            config_local = config["params"].copy()
            config_local["num_cell"] = [i, j]
            sa = SimulatedAnnealing(initial_trees, **config_local)
            trees = sa.solve()
            evaluate_and_store_best(trees)

    def reduce_to_n_trees_inside_plot(tree_list, n):
        """Del farest trees"""
        trees = [tree.clone() for tree in tree_list]

        pts = [Point(float(tree.center_x), float(tree.center_y)) for tree in trees]
        centroid_x = sum(point.x for point in pts) / len(pts)
        centroid_y = sum(point.y for point in pts) / len(pts)
        centroid = Point(centroid_x, centroid_y)

        trees_sorted = sorted(
            trees,
            key=lambda t: Point(float(t.center_x), float(t.center_y)).distance(centroid),
            reverse=True
        )

        return trees_sorted[-n:]

    def fill_missing_solutions(new_trees, max_n=200):
        """Dla plot'ów wybiera najlepsze rozwiązanie z wszystkich dostępnych plotów"""

        available_plots = sorted(new_trees.keys())

        for plot in range(1, max_n):

            # all bigger plots
            biggers_M = [available_plot for available_plot in available_plots if available_plot > plot]

            if not biggers_M:
                print("Brak większego plotu, coś nie tak, ustawić na większą siatkę, albo 100% overlap")
                continue

            best_side = None
            best_trees = None

            # testujemy kazdy wiekszy plot
            for bigger_M in biggers_M:
                _, tree_list = new_trees[bigger_M]
                reduced = reduce_to_n_trees_inside_plot(tree_list, plot)
                side = get_tree_list_side_length(reduced)

                if best_side is None or side < best_side:
                    best_side = side
                    best_trees = reduced
                    best_M = bigger_M

            print(f"Fill N={plot} using best of M={best_M}, side={best_side}")

            # zapisujemy najlepsze
            new_trees[plot] = (best_side, best_trees)
            available_plots.append(plot)
            available_plots.sort()

        return new_trees

    new_trees = fill_missing_solutions(new_trees)

    rows = []
    for n in range(1, 201):
        if n in new_trees:
            side, tree_list = new_trees[n]
            for index_tree, tree in enumerate(tree_list):
                rows.append(
                    {
                        "id": f"{n:03d}_{index_tree}",
                        "x": to_str(tree.center_x),
                        "y": to_str(tree.center_y),
                        "deg": to_str(tree.angle),
                    }
                )

    df = pd.DataFrame(rows)
    save_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv(f"{save_time}_submission.csv", index=False)
    df.to_csv(f"new_submission.csv", index=False)

    print("\n=== PROFILING RESULTS ===")
    for name, stat in sorted(
        FUNC_STATS.items(),
        key=lambda x: x[1]["time"],
        reverse=True
    ):
        avg = stat["time"] / stat["calls"]
        print(
            f"{name:40s} | "
            f"calls={stat['calls']:8d} | "
            f"total={stat['time']:8.2f}s | "
            f"avg={avg*1000:8.3f} ms"
        )


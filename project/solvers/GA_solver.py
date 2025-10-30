from datetime import datetime
import statistics
import random
import copy
import csv
from project.cost import routes_cost
from project.dto import Solution, Instance, Node
from project.constraints import Constraints
from .base import CVRPSolver
from .initializations import init_random, init_greedy


class GACVRPSolver(CVRPSolver):
    def __init__(
            self,
            pop_size=100,
            generations=100,
            Px=0.7,
            Pm=0.2,
            tour_size=5,
            mutation_type="swap",  # "swap" or "inversion"
            selection_type="tournament",  # "tournament" or "roulette"
            initialization_type="random",  # only "random" for now
            population_graph: bool = False,
    ):
        self.pop_size = pop_size
        self.generations = generations
        self.Px = Px
        self.Pm = Pm
        self.tour_size = tour_size
        self.mutation_type = mutation_type
        self.selection_type = selection_type
        self.initialization_type = initialization_type
        self.population_graph = population_graph

    # --- Mutations ---
    def swap_mutation(self, chromosome: list[int]) -> list[int]:
        """mutacja per gen: dla każdego genu z prawdopodobieństwem Pm wykonaj swap z innym genem"""
        for i in range(len(chromosome)):
            if random.random() < self.Pm:
                j = random.randrange(len(chromosome))
                chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
        return chromosome

    def inversion_mutation(self, chromosome: list[int]) -> list[int]:
        """odwracanie losowego fragmentu chromosomu"""
        if random.random() < self.Pm:
            random_index_i, random_index_j = sorted(random.sample(range(len(chromosome)), 2))
            chromosome[random_index_i:random_index_j + 1] = chromosome[random_index_i:random_index_j + 1][::-1]
        return chromosome

    # --- Crossover ---
    def ordered_crossover(self, parent1: list[int], parent2: list[int]) -> list[int]:
        size = len(parent1)
        random_index_i, random_index_j = sorted(random.sample(range(size), 2))
        child = [None for _ in range(size)]
        child[random_index_i:random_index_j + 1] = parent1[random_index_i:random_index_j + 1]

        p2_index = 0
        for id in range(size):
            if child[id] is None:
                while parent2[p2_index] in child:
                    p2_index += 1
                child[id] = parent2[p2_index]
        return child

    # --- Decoding and fitness ---
    def decode_chromosome_to_routes(
            self, chromosome: list[int], instance: Instance, constraints: Constraints
    ) -> list[list[Node]]:
        """Dekoduje chromosom na listę tras z uwzględnieniem pojemności"""
        routes = []
        current_route = []
        current_load = 0
        customer_dict = {customer.id: customer for customer in instance.customers}

        for customer_id in chromosome:
            customer = customer_dict[customer_id]
            if current_load + customer.demand > constraints.truck_capacity:
                if current_route:
                    routes.append(current_route)
                current_route = []
                current_load = 0
            current_route.append(customer)
            current_load += customer.demand

        if current_route:
            routes.append(current_route)

        return routes

    def fitness(self, chromosome: list[int], instance: Instance, constraints: Constraints) -> float:
        depot = instance.depot
        routes = self.decode_chromosome_to_routes(chromosome, instance, constraints)
        solution = routes_cost(routes, depot)
        return solution.total_cost

    # --- Selections ---
    def tournament_selection(self, population: list[list[int]], fitnesses: list[float]) -> list[int]:
        selected = random.sample(list(zip(population, fitnesses)), self.tour_size)
        selected.sort(key=lambda x: x[1])
        return copy.deepcopy(selected[0][0])

    def roulette_selection(self, population: list[list[int]], fitnesses: list[float]) -> list[int]:
        max_fit = max(fitnesses)
        scaled_fitness = [(max_fit - f + 1e-6) for f in fitnesses]
        total = sum(scaled_fitness)
        pick = random.uniform(0, total)

        current = 0
        for chromosome, fit in zip(population, scaled_fitness):
            current += fit
            if current > pick:
                return copy.deepcopy(chromosome)
        return copy.deepcopy(population[-1])

    # --- Strategies ---
    def select_parent_strategy(self, population, fitnesses):
        if self.selection_type == "roulette":
            return self.roulette_selection(population, fitnesses)
        else:
            return self.tournament_selection(population, fitnesses)

    def mutate_strategy(self, chromosome: list[int]) -> list[int]:
        if self.mutation_type == "swap":
            return self.swap_mutation(chromosome)
        else:
            return self.inversion_mutation(chromosome)

    def initialization_strategy(self, instance) -> list[Node]:
        if self.initialization_type == "random":
            return init_random(instance)
        else:
            raise NotImplementedError

    # --- Main GA loop ---
    def solve(self, instance: Instance, constraints: Constraints) -> Solution:
        inited_customers: list[Node] = self.initialization_strategy(instance)
        customers_ids = [customer.id for customer in inited_customers]

        population = [random.sample(customers_ids, len(customers_ids)) for _ in range(self.pop_size)]
        fitnesses = [self.fitness(chromosome, instance, constraints) for chromosome in population]

        best_solution = min(population, key=lambda chrom: self.fitness(chrom, instance, constraints))
        best_cost = self.fitness(best_solution, instance, constraints)

        stats = [] if self.population_graph else None

        for gen in range(1, self.generations + 1):
            new_population = []

            while len(new_population) < self.pop_size:
                parent1 = self.select_parent_strategy(population, fitnesses)
                parent2 = self.select_parent_strategy(population, fitnesses)

                # --- Crossover ---
                if random.random() < self.Px:
                    child = self.ordered_crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)

                # --- Mutation ---
                child = self.mutate_strategy(child)

                new_population.append(child)

            fitnesses = [self.fitness(chrom, instance, constraints) for chrom in new_population]

            gen_best_cost = min(fitnesses)
            gen_worst_cost = max(fitnesses)
            gen_avg_cost = statistics.mean(fitnesses)

            if gen_best_cost < best_cost:
                best_cost = gen_best_cost
                best_solution = copy.deepcopy(new_population[fitnesses.index(gen_best_cost)])

            if self.population_graph:
                stats.append((gen, gen_best_cost, gen_avg_cost, gen_worst_cost))

            population = new_population

        if self.population_graph and stats:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_csv_path = f"GA_population_graph_{timestamp}.csv"
            with open(save_csv_path, "w", newline="") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow(["generation", "best_cost", "avg_cost", "worst_cost"])
                for row in stats:
                    writer.writerow(row)

        best_routes = self.decode_chromosome_to_routes(best_solution, instance, constraints)
        depot = instance.depot
        result = routes_cost(best_routes, depot)

        return Solution(routes=result.routes, total_cost=result.total_cost)

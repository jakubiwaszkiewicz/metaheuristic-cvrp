from datetime import datetime
import csv
import statistics
import random
import math
import copy
from project.constraints import Constraints
from project.dto import Instance, Solution, Node
from project.cost import routes_cost
from .base import CVRPSolver
from .initializations import init_random, init_greedy

class SACVRPSolver(CVRPSolver):
    def __init__(
            self,
            initial_temp: float = 1000.0,
            final_temp: float = 1.0,
            alpha: float = 0.9,
            max_iter_per_temp: int = 100,
            initialization_type="random",
            cooling_step_graph: bool = False,
    ):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.max_iter_per_temp = max_iter_per_temp
        self.initialization_type = initialization_type
        self.cooling_step_graph = cooling_step_graph

    def fitness(self, routes: list[list[Node]], instance: Instance, constraints: Constraints) -> float:
        for route in routes:
            total_load = sum(customer.demand for customer in route)
            if total_load > constraints.truck_capacity:
                return float("inf")

        return routes_cost(routes, instance.depot).total_cost


    def initialization_strategy(self, instance, constraints) -> list[Node]:
        if self.initialization_type == "random":
            return init_random(instance)
        else:
            return init_greedy(instance, constraints)


    def initial_solution(self, instance: Instance, constraints: Constraints) -> list[list[Node]]:
        customers = self.initialization_strategy(instance, constraints)
        routes = []
        current_route = []
        current_load = 0

        for customer in customers:
            if current_load + customer.demand > constraints.truck_capacity:
                routes.append(current_route)
                current_route = []
                current_load = 0
            current_route.append(customer)
            current_load += customer.demand

        if current_route:
            routes.append(current_route)

        return routes

    def move_random_customer_to_available_route(self, routes: list[list[Node]], truck_constraints: Constraints) -> list[
        list[Node]]:
        neighbor_solution = copy.deepcopy(routes)

        route_to_remove_customer_from = random.choice([route for route in neighbor_solution if len(route) > 0])

        customer_to_move = random.choice(route_to_remove_customer_from)
        route_to_remove_customer_from.remove(customer_to_move)

        available_routes_for_customer = [
            route
            for route in neighbor_solution
            if sum(customer.demand for customer in route) + customer_to_move.demand <= truck_constraints.truck_capacity
        ]

        if available_routes_for_customer:
            route_to_insert_customer_into = random.choice(available_routes_for_customer)
        else:
            route_to_insert_customer_into = []
            neighbor_solution.append(route_to_insert_customer_into)

        insert_index = random.randint(0, len(route_to_insert_customer_into))
        route_to_insert_customer_into.insert(insert_index, customer_to_move)

        filtered_non_len_neighbor_solution = [route for route in neighbor_solution if len(route) > 0]

        return filtered_non_len_neighbor_solution


    def solve(self, instance: Instance, constraints: Constraints) -> Solution:
        current_solution = self.initial_solution(instance, constraints)
        current_cost = self.fitness(current_solution, instance, constraints)

        best_solution = copy.deepcopy(current_solution)
        best_cost = current_cost

        temp = self.initial_temp
        temp_step = 0

        stats = [] if self.cooling_step_graph else None

        while temp > self.final_temp:
            step_costs = []

            for _ in range(self.max_iter_per_temp):
                neighbor_solution = self.move_random_customer_to_available_route(current_solution, constraints)
                neighbor_cost = self.fitness(neighbor_solution, instance, constraints)
                step_costs.append(neighbor_cost)

                delta = neighbor_cost - current_cost

                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current_solution = neighbor_solution
                    current_cost = neighbor_cost

                    if current_cost < best_cost:
                        best_solution = copy.deepcopy(current_solution)
                        best_cost = current_cost

            if self.cooling_step_graph:
                best_in_step = min(step_costs)
                worst_in_step = max(step_costs)
                avg_in_step = statistics.mean(step_costs)
                stats.append((temp_step, best_in_step, avg_in_step, worst_in_step))

            temp *= self.alpha
            temp_step += 1

        if self.cooling_step_graph and stats:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_csv_path = f"SA_cooling_step_graph_{timestamp}.csv"

            with open(save_csv_path, mode="w", newline="") as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(["temp_step", "best_cost", "average_cost", "worst_cost"])
                for row in stats:
                    writer.writerow(row + ("", "", ""))

        result = routes_cost(best_solution, instance.depot)
        return Solution(routes=result.routes, total_cost=result.total_cost)

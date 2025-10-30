from project.solvers.base import CVRPSolver
from project.dto import Solution, Instance
from project.cost import routes_cost, distance
from project.constraints import Constraints


class GreedyCVRPSolver(CVRPSolver):
    def solve(self, instance: Instance, constraints: Constraints):
        depot = instance.depot
        all_solutions = []

        # dla każdego klienta rozpocząć konstrukcję rozwiązania od niego
        for start_customer in instance.customers:
            unvisited_customers = set(instance.customers)
            routes = []

            while unvisited_customers:
                load = 0
                route = []
                current_location = depot
                next_start = start_customer if start_customer in unvisited_customers else None

                if next_start:
                    if not constraints.fits_capacity(load, next_start.demand):
                        break
                    route.append(next_start)
                    load += next_start.demand
                    current_location = next_start
                    unvisited_customers.remove(next_start)

                while True:
                    candidates = [
                        customer for customer in unvisited_customers
                        if constraints.fits_capacity(load, customer.demand)
                    ]
                    if not candidates:
                        break

                    next_customer = min(
                        candidates, key=lambda candidate: distance(current_location, candidate)
                    )
                    route.append(next_customer)
                    load += next_customer.demand
                    current_location = next_customer
                    unvisited_customers.remove(next_customer)

                routes.append(route)

            result = routes_cost(routes, depot)
            all_solutions.append(Solution(routes=result.routes, total_cost=result.total_cost))

        return all_solutions

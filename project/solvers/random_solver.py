import random
from project.solvers.base import CVRPSolver
from project.dto import Solution, Instance
from project.cost import routes_cost
from project.constraints import Constraints

class RandomCVRPSolver(CVRPSolver):
    def solve(self, instance: Instance, constraints: Constraints) -> Solution:
        customers = instance.customers[:]
        random.shuffle(customers)

        routes = []
        route_loads = []

        for customer in customers:
            for idx, load in enumerate(route_loads):
                if constraints.fits_capacity(load, customer.demand):
                    routes[idx].append(customer)
                    route_loads[idx] += customer.demand
                    break
            else:
                routes.append([customer])
                route_loads.append(customer.demand)

        return routes_cost(routes, instance.depot)

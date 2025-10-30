import math
from dto import Route, Solution, Node

def distance(node1: Node, node2: Node) -> float:
    dx = node1.x - node2.x
    dy = node1.y - node2.y
    return math.hypot(dx, dy)

def _single_route_cost(route_nodes: list[Node], depot: Node) -> float:
    total_cost = 0.0
    current_node = depot

    for next_node in route_nodes:
        total_cost += distance(current_node, next_node)
        current_node = next_node

    total_cost += distance(current_node, depot)

    return total_cost

def routes_cost(routes: list[list[Node]], depot: Node) -> Solution:
    route_objs = []
    total_cost = 0.0

    for idx, route in enumerate(routes, start=1):
        cost = _single_route_cost(route, depot)
        total_cost += cost
        route_objs.append(Route(route_id=idx, nodes=route, cost=cost))

    return Solution(routes=route_objs, total_cost=total_cost)

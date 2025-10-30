from dataclasses import dataclass

@dataclass
class Node:
    id: int
    x: float
    y: float
    demand: int

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

@dataclass
class Route:
    route_id: int
    nodes: list[Node]
    cost: float
    def __str__(self):
        node_ids = " ".join(str(node.id) for node in self.nodes)
        return f"Route #{self.route_id}: {node_ids} | Cost: {self.cost:.2f}"

@dataclass
class Solution:
    routes: list[Route]
    total_cost: float
    def __str__(self):
        routes_str = "\n".join(str(route) for route in self.routes)
        return f"{routes_str}\nTotal Cost: {self.total_cost:.2f}"

@dataclass
class Instance:
    instance_name: str
    truck_capacity: int
    customers: list[Node]
    depot: Node



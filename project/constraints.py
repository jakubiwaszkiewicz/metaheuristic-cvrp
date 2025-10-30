from dataclasses import dataclass

@dataclass
class Constraints:
    truck_capacity: int
    def fits_capacity(self, current_load: int, customer_demand: int) -> bool:
        return current_load + customer_demand <= self.truck_capacity

from abc import ABC, abstractmethod
from typing import Optional

from project.constraints import Constraints
from project.dto import Instance, Solution


class CVRPSolver(ABC):
    @abstractmethod
    def solve(self, instance: Instance, constraints: Constraints) -> Optional[Solution|list[Solution]]:
        """Solve the CVRP instance and return a Solution"""
        pass



from abc import ABC, abstractmethod


class SimulationEnv:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"SimulationEnv({self.name})"

    @abstractmethod
    def create(self):
        pass

    @abstractmethod
    def simulate(self):
        pass

    @abstractmethod
    def verify(self):
        pass

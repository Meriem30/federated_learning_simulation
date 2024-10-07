import random

from other_libs.log import log_debug


class ClientState:
    def __init__(self, worker_id: int) -> None:
        self._client_id = worker_id
        self._battery: int = 100
        self._energy_consumption: float = 0.0
        self._memory_occupation: float = 0.0
        self._family: int = 1

    def initialize_state(self, families: int) -> None:
        self._battery = random.uniform(50, 100)
        self._energy_consumption = random.uniform(0.1, 1.0)
        self._memory_occupation = random.uniform(0.1, 1.0)
        self._family = random.choice(list(range(1, families)))

    def update_state(self, new_battery: float, new_energy: float, new_memory: float, family: int) -> None:
        self._battery = new_battery
        self._energy_consumption = new_energy
        self._memory_occupation = new_memory
        self._family = family

    @property
    def battery(self) -> float:
        return self._battery

    @property
    def energy_consumption(self) -> float:
        return self._energy_consumption

    @property
    def memory_occupation(self) -> float:
        return self._memory_occupation

    @property
    def family(self):
        return self._family

    def set_family(self, family_id: int):
        self._family = family_id

    def __repr__(self):
        return log_debug(f"ClientState for worker {self._client_id}: "
                         f"(battery={self._battery}, "
                         f"energy_consumption={self._energy_consumption},"
                         f" memory_occupation={self._memory_occupation}, "
                         f"family={self._family})")

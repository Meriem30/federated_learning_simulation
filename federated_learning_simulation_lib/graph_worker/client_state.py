import random

from other_libs.log import log_debug


class ClientState:
    def __init__(self, worker_id: int) -> None:
        self._client_id = worker_id
        self._memory_occupation: float = random.uniform(1,0.1)
        self._energy_consumption: float = random.uniform(1,0.1)
        self._battery: int = random.random()
        self._family: int = 0

    def initialize_state(self, families: int) -> None:
        self._battery = random.uniform(50, 100)
        self._energy_consumption = random.uniform(1.0, 100.0)
        self._memory_occupation = random.uniform(1.0, 1000.0)
        self._family = random.choice(list(range(1, families)))
        log_debug(f"Initialized ClientState: {self}")

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

    @staticmethod
    def get_number_of_properties() -> int:
        return len([attr for attr in vars(ClientState(0)) if
                    not attr.startswith('_client_id') and not attr.startswith('_family')])

    def set_family(self, family_id: int):
        self._family = family_id

    def __repr__(self):
        return (f"{self._client_id}:"
                         f"(battery={self._battery},"
                         f"energy_consumption={self._energy_consumption},"
                         f" memory_occupation={self._memory_occupation},"
                         f"family={self._family})")

    def to_dict(self) -> dict:
        return {
            "client_id": self._client_id,
            "battery": self._battery,
            "energy_consumption": self._energy_consumption,
            "memory_occupation": self._memory_occupation,
            "family": self._family
        }

    def from_dict(self, state_dict: dict) -> None:
        self._client_id = state_dict.get("client_id", self._client_id)
        self._battery = state_dict.get("battery", self._battery)
        self._energy_consumption = state_dict.get("energy_consumption", self._energy_consumption)
        self._memory_occupation = state_dict.get("memory_occupation", self._memory_occupation)
        self._family = state_dict.get("family", self._family)

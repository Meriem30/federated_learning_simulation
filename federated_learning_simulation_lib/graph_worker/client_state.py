import random

from other_libs.log import log_debug


class ClientState:
    def __init__(self, worker_id: int) -> None:
        self._client_id = worker_id
        self._mi: float = 0.0
        self._family: int = 0

    def initialize_state(self, families: int) -> None:
        self._family = random.choice(list(range(1, families)))
        log_debug(f"Initialized ClientState: {self}")

    def update_state(self, new_battery: float, new_energy: float, new_memory: float, new_mi: float, family: int) -> None:
        self._mi = new_mi
        self._family = family

    @property
    def mi(self):
        return self._mi

    @property
    def family(self):
        return self._family

    @staticmethod
    def get_number_of_properties() -> int:
        return len([attr for attr in vars(ClientState(0)) if
                    not attr.startswith('_client_id') and not attr.startswith('_family')])

    def set_family(self, family_id: int) -> None:
        self._family = family_id

    def set_mi(self, mutual_information: float) -> None:
        self._mi = mutual_information

    def __repr__(self):
        return f"worker {self._client_id}: mutual_info: {self._mi:.2f}, family: {self._family}"

    def to_dict(self) -> dict:
        return {
            "client_id": self._client_id,
            "mi": self._mi,
            "family": self._family
        }

    def from_dict(self, state_dict: dict) -> None:
        self._client_id = state_dict.get("client_id", self._client_id)
        self._mi = state_dict.get("mi", self._mi)
        self._family = state_dict.get("family", self._family)

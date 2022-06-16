from abc import ABC, abstractmethod
from typing import Any


class ABSDetector(ABC):
    @abstractmethod
    def detect(self, frame: Any) -> Any:
        pass

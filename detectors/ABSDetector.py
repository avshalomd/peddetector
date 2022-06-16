from abc import ABC, abstractmethod


class ABSDetector(ABC):
    @abstractmethod
    def detect(self, frame):
        pass

from abc import ABC, abstractmethod
from typing import Any


class ABSDetector(ABC):
    """
        An abstract class used to represent humans detector

        Methods
        -------
        detect(frame: Any) -> Any
            Detect human in frame
    """
    @abstractmethod
    def detect(self, frame: Any) -> Any:
        """Detect human in frame

            Parameters
            ----------
            frame : Any
                frame to detect humans in

            Returns
            -------
            Any
                a frame with detected humans in bounding boxes
        """
        pass

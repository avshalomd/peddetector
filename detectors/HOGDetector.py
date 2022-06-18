from typing import Any

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

from detectors import DetectorsConfigs
from detectors.ABSDetector import ABSDetector


class HOGDetector(ABSDetector):
    """
        A class used to represent Histogram of Oriented Gradients humans detector

        Attributes
        ----------
        hog_descriptor : cv2.HOGDescriptor()
            Histogram of Oriented Gradients descriptor used to detect people

        Methods
        -------
        detect(frame: Any) -> Any
            Detect human in frame using HOG people detection descriptor
    """
    def __init__(self):

        self.hog_descriptor = cv2.HOGDescriptor()
        self.hog_descriptor.setSVMDetector(svmdetector=cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: Any) -> Any:
        """Detect human in frame using HOG people detection descriptor

            Parameters
            ----------
            frame : Any
                frame to detect humans in

            Returns
            -------
            Any
                a frame with detected humans in bounding boxes
        """
        bboxes, weights = self.hog_descriptor.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)
        bboxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in bboxes])
        suppressed_detections = non_max_suppression(boxes=bboxes,
                                                    overlapThresh=DetectorsConfigs.SUPPRESSION_OVERLAP_THRESHOLD)
        num_pedestrians = 1
        for x, y, w, h in suppressed_detections:
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(w, h), color=DetectorsConfigs.BBOXES_COLOR_RGB, thickness=2)
            cv2.rectangle(img=frame, pt1=(x, y - 20), pt2=(w, y), color=DetectorsConfigs.BBOXES_COLOR_RGB, thickness=-1)
            cv2.putText(img=frame,
                        text=f'P{num_pedestrians}',
                        org=(x, y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        color=DetectorsConfigs.TEXT_COLOR_RGB,
                        thickness=2)
            num_pedestrians += 1
        return frame

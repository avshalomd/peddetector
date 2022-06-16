import cv2
import numpy as np

from imutils.object_detection import non_max_suppression

from detectors import DetectorsConfigs
from detectors.ABSDetector import ABSDetector


class HOGDetector(ABSDetector):
    def __init__(self):
        # Histogram of Oriented Gradients Detector
        self.hog_descriptor = cv2.HOGDescriptor()
        self.hog_descriptor.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame):
        # Using Sliding window concept
        bboxes, weights = self.hog_descriptor.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.03)
        bboxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in bboxes])
        best_detections = non_max_suppression(boxes=bboxes,
                                              overlapThresh=DetectorsConfigs.NON_MAX_SUPPRESSION_OVERLAP_THRESHOLD)
        num_pedestrians = 1
        for x, y, w, h in best_detections:
            cv2.rectangle(frame, (x, y), (w, h), DetectorsConfigs.BBOXES_COLOR_RGB, 2)
            cv2.rectangle(frame, (x, y - 20), (w, y), DetectorsConfigs.BBOXES_COLOR_RGB, -1)
            cv2.putText(frame, f'P{num_pedestrians}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DetectorsConfigs.TEXT_COLOR_RGB, 2)
            num_pedestrians += 1
        return frame

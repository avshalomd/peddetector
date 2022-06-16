import os

import cv2

import Configs
from detectors.HOGDetector import HOGDetector


def _get_frame_resize(video_capture: cv2.VideoCapture) -> tuple[int, int]:
    orig_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_height = Configs.OUTPUT_FRAME_HEIGHT
    if frame_height <= 0 or frame_height > orig_height:
        return orig_width, orig_height

    frame_ratio = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) / int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_width = int(frame_height // frame_ratio)
    return frame_width, frame_height


def _get_frame_rate(video_capture: cv2.VideoCapture) -> float:
    orig_frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    frame_rate = Configs.OUTPUT_FRAME_RATE
    if frame_rate <= 0:
        return orig_frame_rate
    return frame_rate


def pedestrian_detector(file_path: str, show: bool = True, save: bool = False, output_file_path: str = None):
    video_capture = cv2.VideoCapture(file_path)
    detector = HOGDetector()

    frame_resize = _get_frame_resize(video_capture=video_capture)
    detected_frames = []

    while True:
        ret, frame = video_capture.read()
        if ret:
            detected_frame = detector.detect(cv2.resize(frame, frame_resize))
            detected_frames.append(detected_frame)
            if show:
                cv2.imshow('peddetector', detected_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cv2.destroyAllWindows()

    if save:
        if output_file_path is None:
            file_name = os.path.splitext(os.path.basename(os.path.normpath(file_path)))[0]
            output_file_path = os.path.basename(file_name + '_detected.mp4')

        output_frame_rate = _get_frame_rate(video_capture=video_capture)
        out = cv2.VideoWriter(output_file_path,
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps=output_frame_rate,
                              frameSize=frame_resize)

        for frame in detected_frames:
            out.write(frame)


if __name__ == "__main__":
    my_file_path = os.path.join('data', 'People_Walk.mp4')
    my_output_file_path = os.path.join('output', 'People_Walk_detected.mp4')
    pedestrian_detector(file_path=my_file_path, save=True, output_file_path=my_output_file_path)

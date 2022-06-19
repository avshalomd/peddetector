import os
from typing import Any

import cv2
from tqdm import tqdm

import Configs
from detectors.HOGDetector import HOGDetector


def pedestrians_detector(file_path: str, show: bool = False, save: bool = False, output_file_path: str = None):
    """Detects pedestrians in video

        frame size used by detector and used in saving is calculated using _get_frame_resize() function.
        frame rate used in saving is calculated using  _get_frame_rate() function.

        Parameters
        ----------
        file_path : str
            path to input video file
        show : bool
            show detection on screen in real-time
        save : bool
            save detection output video to file
        output_file_path : str
            path of detection output video file
    """
    video_capture = cv2.VideoCapture(file_path)
    detector = HOGDetector()

    frame_resize = _get_frame_resize(video_capture=video_capture)
    detected_frames = []

    file_name = os.path.basename(os.path.normpath(file_path))
    num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(iterable=range(num_frames), desc=f'Detecting pedestrians in {file_name}', unit='frame'):
        ret, frame = video_capture.read()
        detected_frame = detector.detect(frame=cv2.resize(src=frame, dsize=frame_resize))
        detected_frames.append(detected_frame)
        if show:
            cv2.imshow(winname='peddetector', mat=detected_frame)
        if cv2.waitKey(delay=1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    if save:
        output_frame_rate = _get_frame_rate(video_capture=video_capture)
        if output_file_path is None:
            orig_file_name = os.path.splitext(os.path.basename(os.path.normpath(file_path)))[0]
            output_file_path = os.path.basename(orig_file_name + '_detected.mp4')
        _save_as_video(frames=detected_frames,
                       file_path=output_file_path,
                       frame_size=frame_resize,
                       frame_rate=output_frame_rate)


def _get_frame_resize(video_capture: cv2.VideoCapture) -> tuple[int, int]:
    """Calculate and return frame size for resize according to /Configs.py

        size is calculated such that:
        if Configs.OUTPUT_FRAME_HEIGHT > 0
            frame_height = min(original video frame_height, Configs.OUTPUT_FRAME_HEIGHT)
        else
            frame_height = original video frame_height

        frame_width = int(frame_height / original frame ratio)

        size = (frame_width, frame_height)

        Parameters
        ----------
        video_capture : cv2.VideoCapture
            video file capture

        Returns
        -------
        tuple[int, int]
            frame size for resize
    """
    orig_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_height = Configs.OUTPUT_FRAME_HEIGHT
    if frame_height <= 0 or frame_height > orig_height:
        return orig_width, orig_height

    frame_ratio = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) / video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_width = int(frame_height / frame_ratio)
    return frame_width, frame_height


def _get_frame_rate(video_capture: cv2.VideoCapture) -> float:
    """Calculate and return frame rate according to /Configs.py

        frame rate is calculated such that:
        if Configs.OUTPUT_FRAME_RATE > 0
            frame_rate = Configs.OUTPUT_FRAME_RATE
        else
            frame_rate = original video frame_rate

        Parameters
        ----------
        video_capture : cv2.VideoCapture
            video file capture

        Returns
        -------
        float
            frame rate
    """
    orig_frame_rate = video_capture.get(cv2.CAP_PROP_FPS)
    frame_rate = Configs.OUTPUT_FRAME_RATE
    if frame_rate <= 0:
        return orig_frame_rate
    return frame_rate


def _save_as_video(frames: list[Any], file_path: str, frame_size: tuple[int, int], frame_rate: float):
    """Save a list of frames as video file

        Parameters
        ----------
        frames : list[Any]
            list of frames to save as video
        file_path : str
            path to save output video in
        frame_size : tuple[int, int]
            size of each frame in the output video
        frame_rate : float
            frame rate of output video
    """
    out = cv2.VideoWriter(file_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps=frame_rate,
                          frameSize=frame_size)
    for frame in frames:
        out.write(frame)

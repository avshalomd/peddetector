# peddetector
Pedestrians detector in video.

![detection_example](https://github.com/avshalomd/peddetector/blob/main/data/detection_example.png "Detection example")

## Quick Start
### Windows installation
1. Download and install [python 3.10](https://www.python.org/ftp/python/3.10.5/python-3.10.5-amd64.exe)
2. Download [numpy-1.22.4+mkl](https://www.lfd.uci.edu/~gohlke/pythonlibs/#_numpy) wheel file into whls/
3. `pip install -r requirements.txt`

### Python Usage
```python
import os
from PedestriansDetector import pedestrians_detector

input_file_path = os.path.join('data', 'People_Walk.mp4')
pedestrians_detector(file_path=input_file_path, show=True, save=True)
```

### Script Usage
#### Example
`peddetector.py data/People_Walk.mp4 --show --save`

#### Help Menu
`peddetector.py -h`

## Configs
### General (Configs.py)
`OUTPUT_FRAME_HEIGHT`: frame height used to calculate output video frame size according to original input ratio (int>0 or -1 for original height, e.g. 240/360/720)

`OUTPUT_FRAME_RATE`: output video frame rate (int>0 or -1 for original frame rate, e.g. 24/30/60)
### Detectors (detectors/DetectorsConfigs.py)
`SUPPRESSION_OVERLAP_THRESHOLD`: non-max suppression overlap threshold used by HOGDetector (0.0 < float < 1.0, e.g. 0.3/0.5/0.65)

`BBOXES_COLOR_RGB`: RGB coded color of the detection bounding boxes (tuple[int, int, int], e.g. (20, 200, 20)/(255, 255, 255))

`TEXT_COLOR_RGB`: RGB coded color of the text on the detection bounding boxes (tuple[int, int, int], e.g. (20, 200, 20)/(255, 255, 255))

## TODOs
* docs
* implement Faster-RCNN / YOLO based detectors
* implement evaluation methodology and metrics (accuracy, fps, etc.)
* evaluate multiple detectors models
* unit tests
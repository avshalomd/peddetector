# peddetector
Pedestrian detector in video

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
### General

### Detectors
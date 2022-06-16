import argparse

from PedestriansDetector import pedestrians_detector


def parse_user_arguments():
    parser = argparse.ArgumentParser(description='Pedestrians Detector',
                                     epilog='Usage Example:\n'
                                            'peddetector.py data/People_Walk.mp4 --show --save --output_file out.mp4')
    parser.add_argument('input_file', type=str,
                        help='Input video file path to detect pedestrians in')
    parser.add_argument('--show', action=argparse.BooleanOptionalAction,
                        help='show detection in real-time')
    parser.add_argument('--save', action=argparse.BooleanOptionalAction,
                        help='save detection video to file')
    parser.add_argument('--output_file', type=str, nargs='?',
                        help='Output file path to save the detection video in')
    return parser.parse_args()


args = parse_user_arguments()
pedestrians_detector(file_path=args.input_file, show=args.show, save=args.save, output_file_path=args.output_file)

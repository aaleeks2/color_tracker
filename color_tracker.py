"""
CLI tool for tracking colors on provided video.
Based on MVC pattern.
Libraries involved: opencv-python, numpy, PIL, argparse
"""
from enum import Enum
import argparse
import cv2
import numpy as np
from PIL import Image


WINDOW_NAME = 'Color tracker'
TRACKING_COLOR_ERROR = ('Attempted processing mode that requires a tracking color '
                        'set when no color is set')
NO_COLOR_TRACKER_ERROR = 'Attempted to get frame from uninitialized color tracker.'
KEY_WAIT_TIMEOUT = 10
WHITE_VALUE = 255
BLACK_VALUE = 0

HUE_INDEX = 0
SATURATION_INDEX = 1
VALUE_INDEX = 2


class ProcessingType(Enum):
    """
    Enum describing processing type.
    """
    RAW = 0
    TRACKER = 1
    HUE = 2
    SATURATION = 3
    VALUE = 4
    MASK = 5


# Model
class ColorTracker:
    """
    Class responsible for frame processing.
    Acts as Model part of the MVC pattern.
    """
    _COLOR_RELIANT_PROCESSES = [
        ProcessingType.MASK,
        ProcessingType.TRACKER
    ]

    def __init__(self, video_path: str, hue_tolerance: int, saturation_tolerance: int,
                 value_tolerance: int) -> None:
        self._video = cv2.VideoCapture(video_path)
        if not self._video.isOpened():
            raise FileNotFoundError('Unable to open video file')
        self._tolerances_dict = {
            HUE_INDEX: hue_tolerance,
            SATURATION_INDEX: saturation_tolerance,
            VALUE_INDEX: value_tolerance
        }
        self._tracked_color: None | tuple[int, int, int] = None
        self._frame: None | np.ndarray = None
        self._processed_frame: None | np.ndarray = None
        self._processing_type: ProcessingType = ProcessingType.RAW

    def set_processing_type(self, processing_type: ProcessingType) -> None:
        """
        Setter method fot _processing_type property.
        :param processing_type:
        :return:
        """
        self._processing_type = processing_type

    def set_reference_color(self, x: int, y: int) -> None:
        """
        Setter method fot _tracked_color property.
        :param x:
        :param y:
        :return:
        """
        hsv_frame: np.ndarray = cv2.cvtColor(self._frame, cv2.COLOR_RGB2HSV)
        self._tracked_color = hsv_frame[y, x, :]

    def update_frame(self) -> bool:
        """
        Method responsible for continuous video updating.
        :return:
        """
        read_successful, self._frame = self._video.read()
        if read_successful:
            self._process_frame()
        return read_successful

    def _get_layer(self, layer_index: int) -> np.ndarray:
        """
        Method responsible for retrieving certain layer from HSV-converted frame.
        :param layer_index:
        :return:
        """
        return cv2.cvtColor(self._frame, cv2.COLOR_RGB2HSV)[:, :, layer_index]

    def _process_frame(self) -> None:
        """
        Method responsible for altering current frame based on the selected processing type.
        :return:
        """
        if self._processing_type is ProcessingType.RAW:
            self._processed_frame = self._frame
            return

        hue_layer = self._get_layer(HUE_INDEX)
        saturation_layer = self._get_layer(SATURATION_INDEX)
        value_layer = self._get_layer(VALUE_INDEX)

        merged_mask, frame_with_color_tracking = None, None
        if self._processing_type in self._COLOR_RELIANT_PROCESSES:
            mask_hue = self._get_mask(hue_layer, HUE_INDEX)
            mask_saturation = self._get_mask(saturation_layer, SATURATION_INDEX)
            mask_value = self._get_mask(value_layer, VALUE_INDEX)
            merged_mask = (mask_value & mask_saturation & mask_hue).astype(np.uint8)
            frame_with_color_tracking = self._get_tracked_color_frame(merged_mask)

        if self._processing_type == ProcessingType.HUE:
            self._processed_frame = hue_layer
            return

        if self._processing_type == ProcessingType.SATURATION:
            self._processed_frame = saturation_layer
            return

        if self._processing_type == ProcessingType.VALUE:
            self._processed_frame = value_layer
            return

        if self._processing_type == ProcessingType.MASK:
            self._processed_frame = merged_mask
            return

        if self._processing_type == ProcessingType.TRACKER:
            self._processed_frame = frame_with_color_tracking
            return

    def _get_mask(self, mask: np.ndarray, index: int) -> np.ndarray:
        """
        Method responsible for preparing appropriate value detection in specified layer.
        :param mask:
        :param index:
        :return:
        """
        if self._tracked_color is None:
            raise ValueError(TRACKING_COLOR_ERROR)

        left_inner = np.where(mask <= self._tracked_color[index] + self._tolerances_dict[index],
                              mask, BLACK_VALUE)
        return np.where(self._tracked_color[index] - self._tolerances_dict[index] <= left_inner,
                        WHITE_VALUE,
                        BLACK_VALUE)

    def _get_tracked_color_frame(self, mask: np.ndarray) -> np.ndarray:
        """
        Method responsible for enclosing detected shape in rectangle based on merged HSV masks.
        :param mask:
        :return:
        """
        mask_ = Image.fromarray(mask)
        bbox = mask_.getbbox()

        if bbox is None:
            return mask
        x1, y1, x2, y2 = bbox
        drawing = cv2.rectangle(np.zeros_like(mask, dtype=np.uint8),
                                (x1, y1), (x2, y2), WHITE_VALUE, 2)
        return drawing

    def get_frame(self) -> np.ndarray:
        """
        Method returning current frame of the passed video.
        :return:
        """
        if self._frame is None:
            raise ValueError(NO_COLOR_TRACKER_ERROR)
        return self._frame.copy()

    def get_processed_frame(self) -> np.ndarray:
        """
        Method returning altered current frame.
        :return:
        """
        return self._processed_frame.copy()


# View
class Display:
    """
    Class responsible for displaying frames either original or altered frames from provided video
    Acts as View part of the MVC pattern
    """
    def __init__(self, window_name: str):
        cv2.namedWindow(window_name)
        self._window_name = window_name

    def update_display(self, image: np.ndarray) -> None:
        """
        Method responsible for updating the video with provided frame.
        :param image:
        :return:
        """
        cv2.imshow(self._window_name, image)

    def get_window_name(self) -> str:
        """
        Method returning current window name.
        :return:
        """
        return self._window_name


# Controller
class EventHandler:
    """
    Class responsible for handling communication between ColorTracker (Model) class
    and Display (View) class.
    Acts as Controller part of the MVC pattern.
    """
    PROCESSING_TYPE_BY_KEY = {
        ord('h'): ProcessingType.HUE,
        ord('s'): ProcessingType.SATURATION,
        ord('v'): ProcessingType.VALUE,
        ord('r'): ProcessingType.RAW,
        ord('m'): ProcessingType.MASK,
        ord('t'): ProcessingType.TRACKER
    }

    def __init__(self, tracker: ColorTracker, display: Display, timeout: int) -> None:
        self._window_name = display.get_window_name()
        self._tracker = tracker
        self._timeout = timeout
        cv2.setMouseCallback(self._window_name, self._handle_mouse)

    def _handle_mouse(self, event, x, y, flags=None, params=None) -> None:
        """
        Method responsible for handling mouse clicks.
        :param event:
        :param x:
        :param y:
        :return:
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self._tracker.set_reference_color(x, y)

    def _handle_keys(self) -> bool:
        """
        Method responsible for handling keyboard interrupts.
        :return:
        """
        keycode = cv2.waitKey(self._timeout)
        if keycode == ord('q') or keycode == 27:
            return False
        if keycode in self.PROCESSING_TYPE_BY_KEY:
            self._tracker.set_processing_type(self.PROCESSING_TYPE_BY_KEY[keycode])
        return True

    def handle_events(self) -> bool:
        """
        Public method responsible handling for keyboard interrupts.
        :return:
        """
        return self._handle_keys()


def parse_arguments() -> argparse.Namespace:
    """
    Method responsible for parsing command line arguments.
    :return:
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-v', '--video-path', required=True, type=str,
                        help='Input video file')
    parser.add_argument('--hue-tolerance', type=str, default=5, required=False,
                        help='Hue tolerance, defaults to 5')
    parser.add_argument('--saturation-tolerance', type=str, default=50, required=False,
                        help='Saturation tolerance. defaults to 50')
    parser.add_argument('--value-tolerance', type=str, default=50, required=False,
                        help='Value tolerance defaults to 50')

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """
    Main method responsible for running the main functionality.
    :param args:
    :return:
    """
    try:
        tracker = ColorTracker(args.video_path, args.hue_tolerance, args.saturation_tolerance,
                               args.value_tolerance)
        display = Display(WINDOW_NAME)
        event_handler = EventHandler(tracker, display, KEY_WAIT_TIMEOUT)
        while True:
            if not tracker.update_frame():
                break
            display.update_display(tracker.get_processed_frame())
            if not event_handler.handle_events():
                break

    except ValueError as e:
        print(e)
    except FileNotFoundError as e:
        print(e)


if __name__ == '__main__':
    main(parse_arguments())

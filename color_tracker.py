# pylint: disable=no-member
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


class PreprocessingType(Enum):
    """
    Enum describing preprocessing type.
    """
    ERODE = 0
    DILATE = 1


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
        self._tracked_color: None | tuple[int, int, int] = None
        self._frame: None | np.ndarray = None
        self._processed_frame: None | np.ndarray = None
        self._processing_type: ProcessingType = ProcessingType.RAW
        self._erosion_depth: int = 0
        self._dilation_depth: int = 0

        self._tolerances_dict = {
            HUE_INDEX: hue_tolerance,
            SATURATION_INDEX: saturation_tolerance,
            VALUE_INDEX: value_tolerance
        }
        self._preprocessing_depth_handler = {
            PreprocessingType.ERODE: self._increment_erode_depth,
            PreprocessingType.DILATE: self._increment_dilate_depth
        }
        self._processing_orchestrator = {
            ProcessingType.RAW: self._get_current_frame,
            ProcessingType.HUE: self._prepare_hue_layer,
            ProcessingType.SATURATION: self._prepare_saturation_layer,
            ProcessingType.VALUE: self._prepare_value_layer,
            ProcessingType.MASK: self._prepare_mask_layer,
            ProcessingType.TRACKER: self._prepare_tracking_layer
        }

    def get_processed_frame(self) -> np.ndarray:
        """
        Method returning altered current frame.
        :return: a copy of processed frame
        """
        temp = self._processed_frame.copy()
        return temp

    def set_processing_type(self, processing_type: ProcessingType) -> None:
        """
        Setter method fot _processing_type property.
        :param processing_type:
        :return: None
        """
        self._processing_type = processing_type

    def set_reference_color(self, x: int, y: int) -> None:
        """
        Setter method fot _tracked_color property.
        :param x: X coordinate of cursor
        :param y: Y coordinate of cursor
        :return: None
        """
        hsv_frame: np.ndarray = cv2.cvtColor(self._frame, cv2.COLOR_RGB2HSV)
        self._tracked_color = hsv_frame[y, x, :]

    def update_ed_depths(self, preprocess: PreprocessingType) -> None:
        """
        Handles preprocessing depths changes.
        :param preprocess: Preprocessing type
        :return: None
        """
        self._preprocessing_depth_handler[preprocess]()

    def update_frame(self) -> bool:
        """
        Method responsible for continuous video updating.
        :return: bool: True if video was read successfully, False otherwise.
        """
        read_successful, self._frame = self._video.read()
        if read_successful:
            self._preprocess_frame()
            self._process_frame()
        return read_successful

    def _increment_erode_depth(self) -> None:
        """
        Increment the erosion depth variable.
        :return: None
        """
        self._erosion_depth += 1
        print(f'Erode depth: {self._erosion_depth}')

    def _increment_dilate_depth(self) -> None:
        """
        Increments the dilation depth variable.
        :return: None
        """
        self._dilation_depth += 1
        print(f'Dilate depth: {self._dilation_depth}')

    def _erode(self, img: np.ndarray) -> np.ndarray:
        """
        Method responsible for applying erosion to image passed as a parameter.
        Uses (5, 5) kernel for optimal application.
        :param img: np.ndarray: image submitted for erosion
        :return: np.ndarray: eroded image
        """
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(img, kernel, iterations=self._erosion_depth)

    def _dilate(self, img: np.ndarray) -> np.ndarray:
        """
        Method responsible for applying dilatation to image passed as a parameter.
        Uses (5, 5) kernel for optimal application.
        :param img: np.ndarray: image submitted for dilatation
        :return: np.ndarray: dilated image
        """
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(img, kernel, iterations=self._dilation_depth)

    def _validate_color(self) -> None:
        """
        Method responsible for perform None-check on _tracked_color property.
        Throws ValueError when _tracked_color is set to None.
        :return: None
        """
        if self._tracked_color is None:
            raise ValueError(TRACKING_COLOR_ERROR)

    def _get_current_frame(self) -> np.ndarray:
        return self._frame

    def _get_layer(self, layer_index: int) -> np.ndarray:
        """
        Method responsible for retrieving certain layer from HSV-converted frame.
        :param layer_index: index of the layer (Hue:0, Saturation:1, Value:2)
        :return: layer of the HSV frame color
        """
        return cv2.cvtColor(self._frame, cv2.COLOR_RGB2HSV)[:, :, layer_index]

    def _get_mask(self, mask: np.ndarray, index: int) -> np.ndarray:
        """
        Method responsible for preparing appropriate value detection in specified layer.
        :param mask: layer of the HSV frame color
        :param index: index of the layer (Hue:0, Saturation:1, Value:2)
        :return: black/white result mask based on the tolerance and selected color
        """
        related_tolerance = self._tolerances_dict[index]
        left_inner = np.where(mask <= self._tracked_color[index] + related_tolerance,
                              mask,
                              BLACK_VALUE)

        return np.where(self._tracked_color[index] - related_tolerance <= left_inner,
                        WHITE_VALUE,
                        BLACK_VALUE)

    def _get_tracked_color_frame(self, mask: np.ndarray) -> np.ndarray:
        """
        Method responsible for enclosing detected shape in rectangle based on merged HSV masks.
        Bounding box consists of 4 coordinates (x1, y1, x2, y2) that are marking corners of the
        bounding box. None-check performed to handle situations -
        - when the shape is absent on the screen.
        :param mask: merged H, S and V masks to one mask
        :return: frame with framed object of a certain color
        """
        mask_image = Image.fromarray(mask)
        bounding_box = mask_image.getbbox()

        if bounding_box is None:
            return mask
        x1, y1, x2, y2 = bounding_box
        drawing = cv2.rectangle(np.zeros_like(mask, dtype=np.uint8),
                                (x1, y1), (x2, y2), WHITE_VALUE, 2)
        return drawing

    def _prepare_hue_layer(self) -> np.ndarray:
        """
        Method responsible for retrieving hue layer of HSV frame
        :return: np.ndarray: Hue layer HSV frame
        """
        return self._get_layer(HUE_INDEX)

    def _prepare_saturation_layer(self) -> np.ndarray:
        """
        Method responsible for retrieving saturation layer of HSV frame
        :return: np.ndarray: Saturation layer of HSV frame
        """
        return self._get_layer(SATURATION_INDEX)

    def _prepare_value_layer(self) -> np.ndarray:
        """
        Method responsible for retrieving value layer of HSV frame
        :return: np.ndarray: Value layer of HSV frame
        """
        return self._get_layer(VALUE_INDEX)

    def _prepare_mask_layer(self) -> np.ndarray:
        """
        Method responsible for creating color specific mask layer, consisting of all hue,
        saturation and value layers.
        :return: np.ndarray: Merged color-specific mask
        """
        self._validate_color()
        mask_hue = self._get_mask(self._prepare_hue_layer(), HUE_INDEX)
        mask_saturation = self._get_mask(self._prepare_saturation_layer(), SATURATION_INDEX)
        mask_value = self._get_mask(self._prepare_value_layer(), VALUE_INDEX)
        return (mask_value & mask_saturation & mask_hue).astype(np.uint8)

    def _prepare_tracking_layer(self) -> np.ndarray:
        """
        Method responsible for creating color specific mask layer, consisting of all hue,
        saturation and value layers.
        :return: np.ndarray: Merged color-specific mask
        """
        return self._get_tracked_color_frame(self._prepare_mask_layer())

    def _preprocess_frame(self) -> None:
        """
        Method responsible for preprocessing the frame - apply erosion and dilatation accordingly to
        private depth variables incremented by user.
        :return:
        """
        eroded = self._erode(self._frame)
        eroded_and_dilated = self._dilate(eroded)
        self._frame = eroded_and_dilated

    def _process_frame(self) -> None:
        """
        Method responsible for altering current frame based on the selected processing type.
        :return: None
        """
        self._processed_frame = self._processing_orchestrator[self._processing_type]()


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
        :param image: original or altered video frame
        :return: None
        """
        cv2.imshow(self._window_name, image)

    def get_window_name(self) -> str:
        """
        Method returning current window name.
        :return: str: window name
        """
        return self._window_name


class EventHandler:
    """
    Class responsible for handling communication between ColorTracker (Model) and Display (View).
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

    PREPROCESSING_TYPE_BY_KEY = {
        ord('e'): PreprocessingType.ERODE,
        ord('d'): PreprocessingType.DILATE
    }

    def __init__(self, tracker: ColorTracker, display: Display, timeout: int) -> None:
        self._window_name = display.get_window_name()
        self._tracker = tracker
        self._timeout = timeout
        cv2.setMouseCallback(self._window_name, self._handle_mouse)

    def _handle_mouse(self, event, x, y, flags=None, params=None) -> None:
        """
        Method responsible for handling mouse clicks.
        :param event:event object
        :param x: x coordinate of the cursor collected after click
        :param y: y coordinate of the cursor collected after click
        :return: None
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self._tracker.set_reference_color(x, y)

    def _handle_keys(self) -> bool:
        """
        Method responsible for handling keyboard interrupts.
        :return: bool: False if quit key is pressed, True otherwise
        """
        keycode = cv2.waitKey(self._timeout)
        if keycode == ord('q') or keycode == 27:
            return False
        if keycode in self.PROCESSING_TYPE_BY_KEY:
            self._tracker.set_processing_type(self.PROCESSING_TYPE_BY_KEY[keycode])
        elif keycode in self.PREPROCESSING_TYPE_BY_KEY:
            self._tracker.update_ed_depths(self.PREPROCESSING_TYPE_BY_KEY[keycode])
        return True

    def handle_events(self) -> bool:
        """
        Public method responsible handling for keyboard interrupts.
        :return: result of handling keyboard interrupts
        """
        return self._handle_keys()


def parse_arguments() -> argparse.Namespace:
    """
    Method responsible for parsing command line arguments.
    :return: argparse.Namespace containing parsed arguments
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
    :param args: parsed command line arguments
    :return: None
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

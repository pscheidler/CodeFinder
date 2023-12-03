import json
import string
from cv2 import boundingRect
from typing import Sequence, Optional, List, Any
import dataclasses

class ContourNotFound(Exception):
    def __init__(self, message="Contour not found"):
        self.message = message
        super().__init__(self.message)


@dataclasses.dataclass
class ContourElement:
    contour: Any = None
    box: Optional[Sequence[int]] = None
    classification: str = "unc"
    active: bool = False
    group: int = 0


class ContourContainer:
    """Container to look up contour rectangles by classification or point

    Contours are stored as x, y, w, h. X, Y of the bottom left point, w = width, h = height
    Goal is to have an initial, simple implementation, and then add in tests and more efficiency
    """
    contours: List[ContourElement]

    def __init__(self, width: int=806, height: int=504, min_width: int=0, min_height: int=0 ):
        self.contours = []
        self.min_size = [min_width, min_height]
        self.max_group = 0

    def add(self, contour, classification="unc"):
        assert(classification == "unc" or classification in string.ascii_lowercase,
               f"Illegal classification of {classification}")
        rect = boundingRect(contour)
        if rect[2] < self.min_size[0] or rect[3] < self.min_size[1]:
            print(f"Contour is too small")
            return
        self.contours.append(ContourElement(box=rect, contour=contour, classification=classification))

    def remove(self, index: int):
        assert(index < len(self.contours), f"Subtracting illegal index of {index} from contours length {len(self.contours)}")
        del self.contours[index]

    @staticmethod
    def point_in_rect(x_in: int, y_in: int, rect: Sequence[int]) -> bool:
        x, y, w, h = rect
        return x <= x_in <= x + w and y <= y_in <= y + h

    def get_index_by_point(self, x_in: int, y_in: int) -> int:
        """
        Find the matchin contour box that contains a point
        :params x_in, y_in: (x, y) point position
        :return: Contour index or -1 for Not Found
        """
        for index, contour in enumerate(self.contours):
            if self.point_in_rect(x_in, y_in, contour.box):
                return index
        return -1

    def get_box(self, x_in: int, y_in: int) -> Sequence[int]:
        index = self.get_index_by_point(x_in, y_in)
        if index < 0:
            raise ContourNotFound(message=f"No contour found at {x_in}, {y_in}")
        return self.contours[index].box

    def classify(self, classification: string="unc", index: Optional[int]=None, x_in: Optional[int]=None,
                 y_in: Optional[int]=None, all=False) -> None:
        if all:
            for contour in self.contours:
                contour.classification = classification
            return
        if index is None:
            if x_in is None or y_in is None:
                assert(False, f"Classify called with no parameters")
            index = self.get_index_by_point(x_in, y_in)
        self.contours[index].classification = classification

    def get_boxes(self, classification: Optional[str]=None, active: Optional[bool]=None, all: bool=False):
        if all:
            for contour in self.contours:
                yield contour.box
            return
        if active is not None:
            for box in [x.box for x in self.contours if x.active==active]:
                yield box
            return
        if classification is not None:
            for box in [x.box for x in self.contours if x.classification == classification]:
                yield box

    def save(self, filename: str):
        with open(filename, 'w+') as fp:
            fp.write(json.dumps(self.__dict__))

    def load(self, filename: str):
        with open(filename, 'r') as fp:
            output = json.load(fp)
        print(output)

    def select_boxes(self, x_in: int=None, y_in: int=None):
        index = self.get_index_by_point(x_in, y_in)
        self.contours[index].active = True

    def unselect_boxes(self, x_in: int=None, y_in: int=None, all=False):
        if all:
            for contour in self.contours:
                contour.active = False
            return
        index = self.get_index_by_point(x_in, y_in)
        self.contours[index].active = False

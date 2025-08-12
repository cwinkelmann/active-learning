"""
collection of type which should help with my annotations, bounding boxes etc.

"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import shapely
from shapely import Polygon


@dataclass(frozen=True)
class Bbox:
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass(frozen=True)
class BboxXYXY:
    x1: int
    y1: int
    x2: int
    y2: int

    def to_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class BboxXYWH:
    """
    center and width and height
    """
    xc: int
    yc: int
    w: int
    h: int

    def to_list(self) -> list[int]:
        return [self.xc, self.yc, self.w, self.h]


@dataclass(frozen=True)
class BboxOBBXY:
    """
    oriented bounding box
    """
    x1: int
    y1: int
    x2: int
    y2: int
    x3: int
    y3: int
    x4: int
    y4: int

    def to_polygon(self) -> shapely.Polygon:
        return Polygon([(self.x1, self.y1), (self.x2, self.y2), (self.x3, self.y3), (self.x4, self.y4)])

    def to_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2, self.x3, self.y3, self.x4, self.y4]

def bbox_creator(row: pd.Series, output_path: Path):
    s = pd.Series((row["attributes"]))
    x1, y1, x2, y2 = row["bbox"]
    poly = pd.Series({"bbox_polygon": Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])})
    return pd.concat([row, s, poly])


def bbox_x_y_wh_creator(xyxy: [int]):
    """
    :param xyxy: list of integer x1 y1 x2 y2
    :return: Bounding box with center x, center y and the width
    """

    x1 = xyxy[0]
    y1 = xyxy[1]
    x2 = xyxy[2]
    y2 = xyxy[3]

    assert x1 < x2
    assert y1 < y2

    h = abs(y2 - y1)
    w = abs(x2 - x1)
    xc = x1 + int(round((w / 2)))
    yc = y1 + int(round((h / 2)))

    return BboxXYWH(
        xc=xc,
        yc=yc,
        w=w,
        h=h,
    )


def bboxXYXYCreator(xyxy: [int]):
    """
    two corners
    :param xyxy: list of integer x1 y1 x2 y2
    :return:
    """
    return BboxXYXY(
        x1=xyxy[0],
        y1=xyxy[1],
        x2=xyxy[2],
        y2=xyxy[3],
    )


@dataclass(frozen=True)
class BboxX1Y1WH:
    """
    corner x and y and width
    """
    x1: int
    y1: int
    w: int
    h: int

    def to_list(self) -> list[int]:
        return [self.x1, self.y1, self.w, self.h]


def bboxX1Y1WHCreator(xyxy: [int]):
    """
    :param xyxy: list of integer x1 y1 x2 y2
    :return:
    """
    x1, y1, x2, y2 = xyxy
    x1 = xyxy[0]
    y1 = xyxy[1]
    x2 = xyxy[2]
    y2 = xyxy[3]

    assert x1 < x2
    assert y1 < y2

    h = abs(y2 - y1)
    w = abs(x2 - x1)

    return BboxX1Y1WH(
        x1=x1,
        y1=y1,
        w=w,
        h=h,
    )


def xyxy2xywh(xyxy: BboxXYXY) -> BboxX1Y1WH:
    """
    :param xyxy:
    :return:
    """
    return BboxX1Y1WH(
        x1=xyxy.x1,
        y1=xyxy.y1,
        w=abs(xyxy.x2 - xyxy.x1),
        h=abs(xyxy.y2 - xyxy.y1)
    )


def xywh2xyxy(xywh: BboxXYWH) -> BboxXYXY:
    """
    :param xywh:
    :return:
    """
    return BboxXYXY(
        x1=xywh.xc - int(round(xywh.w / 2)),
        y1=xywh.yc - int(round(xywh.h / 2)),
        x2=xywh.xc + int(round(xywh.w / 2)),
        y2=xywh.yc + int(round(xywh.h / 2)),
    )


@dataclass(frozen=True)
class Annotation:
    """
    Bounding box annotation class
    """
    class_id: int
    image_name: str
    bbox: BboxXYXY


def annotation_creator(df: pd.DataFrame) -> list[Annotation]:
    lst = df.apply(lambda x: Annotation(
        class_id=x["class_name"],
        bbox=bboxXYXYCreator(x["bbox"]),
        image_name=x["image_name"]),
                   axis=1
                   )

    return lst


@dataclass(frozen=True)
class LargeAnnotation:
    """

    """
    ID: int
    class_name: str
    centroid: shapely.Point
    image_name: str
    bbox: BboxXYXY
    bbox_polygon: shapely.Polygon


def Large_annotation_creator(df: pd.DataFrame) -> list[LargeAnnotation]:
    lst = df.apply(lambda x: LargeAnnotation(
        ID=int(x["ID"]) if "ID" in x else None,
        class_name=x["class_name"],
        bbox=bboxXYXYCreator(x["bbox"]),
        bbox_polygon=x["bbox_polygon"],
        image_name=x["image_name"],
        centroid=x["centroid"]
    ),
                   axis=1)

    return list(lst)

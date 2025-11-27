from enum import Enum


class SampleStrategy(Enum):
    FIRST = 1
    RANDOM = 2
    PERCENTAGE = 3
    ORDERED_ASC = 4
    ORDERED_DESC = 5


class SpatialSampleStrategy(Enum):
    NEAREST = 1
    RANDOM = 2

import copy

import random

import abc

from com.biospheredata.types.HastyAnnotationV2 import HastyAnnotationV2


class CVATAnnotationAdapter():
    """ TODO implement this to create seamless interaction with cvat """
    def __init__(self, hA: HastyAnnotationV2):
        self.hA = hA



class ImageFilter():
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, hA: HastyAnnotationV2):
        return hA


class ImageFilterConstantNum(ImageFilter):
    """ Get N images randomly

    """

    def __init__(self, num: int, seed: int = 42):
        super().__init__()
        self.num = num

        random.seed(seed)

    def __call__(self, hA: HastyAnnotationV2):
        """ get N images randomly
        :return:
        """
        hA = super().__call__(hA)
        if self.num is None:
            return hA
        if self.num > len(hA.images):
            raise ValueError(f"Number of images is {len(hA.images)} but you requested {self.num}")
        sample = random.sample(hA.images, self.num)

        hA_filtered = copy.deepcopy(hA)

        hA_filtered.images = sample
        return hA_filtered
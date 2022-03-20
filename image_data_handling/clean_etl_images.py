import numpy.typing as npt
from skimage.morphology import skeletonize


def clean_image(image: npt.ArrayLike) -> npt.ArrayLike:
    threshold_image = image < 255
    skeletonized_image = skeletonize(threshold_image)

    return skeletonized_image

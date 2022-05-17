import collections
import math

import cv2
import numpy


def vectorLength(p_1, p_2) -> int:
    return round(math.sqrt((p_1[0] - p_2[0]) ** 2 + (p_1[1] - p_2[1]) ** 2))


def unique_count_app(a):
    colors, count = numpy.unique(a.reshape(-1, 3), axis=0, return_counts=True)
    return colors[count.argmax()]

    #colors, count = numpy.unique(a.reshape(-1, 3), axis=0, return_counts=True)
    #return


def findCorners():
    """find corner for deformation"""
    pass


def deformImage(image: numpy.ndarray, polygon: list) -> numpy.ndarray:
    """deform image"""
    #y, x = image.shape[:2]
    x = vectorLength(polygon[0], polygon[1])
    y = vectorLength(polygon[1], polygon[2])
    source = numpy.float32(polygon)
    target = numpy.float32([[0, 0], [x, 0], [x, y], [0, y]])
    matrix = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(image, matrix, (x, y))

def colorCorrection(image: numpy.ndarray):
    """correct colors"""

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = unique_count_app(image)[2]

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            image[y, x, 2] = min(255, image[y, x, 2] / value * 255)




    #
    #
    #
    # color = unique_count_app(image)
    #
    # image[:, :, 2] *= 255 / color[2]
    #
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image


    #mini_image = cv2.resize(image, (100, 100)) // 16
    #colors = collections.defaultdict(int)
    # for y in range(100):
    #     for x in range(100):
    #         color = tuple(mini_image[x, y])
    #         colors[color] += 1


    #color = numpy.array([[max(colors.items(), key=lambda c: c[1])[0]]], dtype="uint8") * 16
    #print(color.shape)
    ##color_canvas[0, 0, :] = color
    #hue = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[0, 0, 0]


def sharpening2(image, kernel_size=(2, 2), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = numpy.maximum(sharpened, numpy.zeros(sharpened.shape))
    sharpened = numpy.minimum(sharpened, 255 * numpy.ones(sharpened.shape))
    sharpened = sharpened.round().astype(numpy.uint8)
    if threshold > 0:
        low_contrast_mask = numpy.absolute(image - blurred) < threshold
        numpy.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def sharpening(image):
    """sharpen image"""
    kernel = numpy.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

    pass


if __name__ == '__main__':
    im = cv2.imread("D:/Screenshot_1.png")
    im = sharpening(im)
    im = colorCorrection(im)

    cv2.imshow("windows", im)
    cv2.waitKey()

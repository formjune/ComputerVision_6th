import math
import os.path
import cv2
import numpy
import numpy as np


def getDistance(p_0, p_1) -> float:
    """distance between 2 points"""
    return math.sqrt((p_0[0][0] - p_1[0][0]) ** 2 + (p_0[0][1] - p_1[0][1]) ** 2)


def getCenters(points) -> float:
    """distance between diagonal centers"""
    p_1 = (points[0][0] + points[2][0]) / 2.0, (points[0][1] + points[2][1]) / 2.0
    p_2 = (points[1][0] + points[3][0]) / 2.0, (points[1][1] + points[3][1]) / 2.0
    return getDistance([p_1], [p_2])


def reorderVertices(points, angle_x, angle_y):
    """find the closest to top left corner vertex"""

    distance = math.inf
    index = 0
    for i, point in enumerate(points):
        new_distance = getDistance(point, [[angle_x, angle_y]])
        if new_distance < distance:
            distance = new_distance
            index = i

    if not index:
        return points
    points = list(points)
    return points[index:] + points[:index]


def mergePoints(points) -> list:
    """merge points until there are 4 left"""
    points = list(points)
    while len(points) > 4:
        distance = math.inf
        skip_point = None
        for i, p_0 in enumerate(points):
            j = (i + 1) % len(points)
            new_distance = getDistance(p_0, points[j])
            if new_distance < distance:
                distance = new_distance
                skip_point = j

        points.pop(skip_point)
    return points


def proceedImage(folder: str, file: str, is_otsu: bool = False) -> tuple:
    # read image
    filename = os.path.join(folder, file)
    img = cv2.imread(filename)
    assert img is not None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    if is_otsu:
        thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    else:
        thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)[1]

    # remove small details to get simplified shape
    kernel = np.ones((7, 7), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

    # get largest contour
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    area_thresh = 0
    big_contour = None
    for c in contours:
        area = cv2.contourArea(c)
        if area > area_thresh:
            area_thresh = area
            big_contour = c

    # get perimeter and contours
    assert big_contour is not None
    peri = cv2.arcLength(big_contour, True)
    corners = cv2.approxPolyDP(big_contour, 0.04 * peri, True)

    # find rectangle and warp
    corners = reorderVertices(corners, img.shape[1], 0)
    if len(corners) != 4:
        corners = mergePoints(corners)
    assert len(corners) == 4
    width = 0.5 * (corners[0][0][0] - corners[1][0][0] + corners[3][0][0] - corners[2][0][0])
    height = 0.5 * (corners[2][0][1] - corners[1][0][1] + corners[3][0][1] - corners[0][0][1])
    width = np.int0(width)
    height = np.int0(height)

    in_corner = np.float32([corner[0] for corner in corners])
    # draw_contour = numpy.reshape(in_corner, (-1, 1, 2)).astype("int32")
    # cv2.drawContours(img, big_contour, -1, (0, 0, 255), 3)
    # cv2.circle(img, in_corner[0].astype("uint32"), 5, (255, 0, 0), -1)
    # cv2.circle(img, in_corner[1].astype("uint32"), 5, (0, 255, 0), -1)
    # cv2.circle(img, in_corner[2].astype("uint32"), 5, (0, 0, 255), -1)
    # cv2.circle(img, in_corner[3].astype("uint32"), 5, (255, 255, 0), -1)

    out_corner = [[width, 0], [0, 0], [0, height], [width, height]]
    out_corner = np.float32(out_corner)
    matrix = cv2.getPerspectiveTransform(in_corner, out_corner)
    return getCenters(in_corner), (img, matrix, (width, height))


def main(folder_in: str, folder_out: str, light_up: bool = False, resize: float = 1.0):
    for file in os.listdir(folder_in):
        try:
            dis_1, args_1 = proceedImage(folder_in, file, False)
        except AssertionError:
            dis_1, args_1 = math.inf, None

        try:
            dis_2, args_2 = proceedImage(folder_in, file, True)
        except AssertionError:
            dis_2, args_2 = math.inf, None

        if args_1 is None and args_2 is None:
            continue
        args = args_1 if dis_1 < dis_2 else args_2
        # image = args[0]
        image = cv2.warpPerspective(*args)

        # add histogram
        if light_up:
            image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            max_color = numpy.max(image_yuv[:, :, 0])
            light_channel = image_yuv[:, :, 0] * (255 / max_color)
            image_yuv[:, :, 0] = light_channel.astype("uint8")
            image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

        if resize != 1.0:
            image = cv2.resize(image, (image.shape[1] * resize, image.shape[0] * resize))

        cv2.imwrite(os.path.join(folder_out, file), image)


if __name__ == '__main__':
    main(r"D:\test_files", r"D:\test_files_out", True)

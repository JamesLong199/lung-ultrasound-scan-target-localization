import cv2 as cv
import math
import numpy as np


def annotate_tag(result, undistorted_img):
    """
        Annotate on one tag on the undistorted image
        - result: one AprilTag detection result
        - undistorted_img: the image to be annotated
    """
    (ptA, ptB, ptC, ptD) = result.corners
    ptB = (int(ptB[0]), int(ptB[1]))
    ptC = (int(ptC[0]), int(ptC[1]))
    ptD = (int(ptD[0]), int(ptD[1]))
    ptA = (int(ptA[0]), int(ptA[1]))

    # draw the bounding box of the AprilTag detection
    cv.line(undistorted_img, ptA, ptB, (0, 255, 0), 4)
    cv.line(undistorted_img, ptB, ptC, (0, 255, 0), 4)
    cv.line(undistorted_img, ptC, ptD, (0, 255, 0), 4)
    cv.line(undistorted_img, ptD, ptA, (0, 255, 0), 4)

    # draw the center (x, y)-coordinates of the AprilTag
    (cX, cY) = (int(result.center[0]), int(result.center[1]))
    cv.circle(undistorted_img, (cX, cY), 5, (0, 0, 255), -1)

    # draw the tag family on the image
    tagFamily = result.tag_family.decode("utf-8")
    cv.putText(undistorted_img, tagFamily, (ptA[0], ptA[1] + 50),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

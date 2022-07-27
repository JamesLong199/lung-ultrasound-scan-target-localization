# Given target 2D coordinates and camera matrices from two camera views,
# triangulate the target 3D coordinates

import numpy as np
import cv2 as cv


def to_homog(points):
    """
    Function: convert points from Euclidean coordinates to homogeneous coordinates
    points: 3xn numpy array containing Euclidean coordinates
    Return: 4xn numpy array containing homogeneous coordinates
    """
    m, n = points.shape
    points_homog = np.concatenate([points, np.ones([1, n])], axis=0)
    return points_homog


def from_homog(points_homog):
    """
    Function: convert points from homogeneous coordinates to Eulidean coordinates
    points_homog: 4xn numpy array containing homogeneous coordinates
    Return: 3xn numpy array containing Euclidean coordinates
    """
    m, n = points_homog.shape
    points = points_homog[:m-1] / points_homog[m-1]
    return points


def reconstruct(pts1, pts2, int1, int2, ext1, ext2):
    """
   Function: reconstruct 3D points with given correspondence
   int1, int2: intrinsic matrices of camera 1 and camera 2
   ext1, ext2: extrinsic matrices of camera 1 and camera 2
   Return: 3xn numpy arrays containing the Euclidean coordinates of reconstructed 3D points
   """

    I_0 = np.hstack((np.eye(3), np.zeros((3, 1))))
    proj1 = int1 @ I_0 @ ext1
    proj2 = int2 @ I_0 @ ext2

    homo_coor = cv.triangulatePoints(proj1, proj2, pts1, pts2)

    recon = from_homog(homo_coor)
    return recon



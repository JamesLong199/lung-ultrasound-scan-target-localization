# Two functions
# convert (rotation matrix,translation vector) to (global 3d position, euler angles)
# convert (global 3d position, euler angles) to (rotation matrix,translation vector)
# Used functions from 'Rotation Matrix To Euler Angles' by Satya Mallick
# https://learnopencv.com/rotation-matrix-to-euler-angles/
import math
import numpy as np

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :
    # inputs
    # -- theta: tuple/list (theta_x,theta_y,theta_z) euler angles (in radians)

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

# convert (rotation matrix,translation vector) to (global 3d position, euler angles)
# output is the pose at the center, which is used as the origin of its own frame
# e.g. tag center of apriltag
def transformation_to_pose(transformation):
    # input
    # -- transformation: tuple/list (3x3 rotation matrix R, 3x1 translation vector t)
    # output
    # -- pose: tuple (global position (x,y,z), euler angles (theta_x,theta_y,theta_z))
    assert isinstance(transformation,tuple) or isinstance(transformation,list)
    assert len(transformation) == 2
    R, t = transformation
    assert R.shape == (3,3) and t.size == 3
    pos = t.reshape(3,)
    angles = rotationMatrixToEulerAngles(R)

    return (pos,angles)


# convert (global 3d position, euler angles) to (rotation matrix,translation vector)
# must be the pose data at the center, which is used as the origin of its own frame
def pose_to_transformation(pose):
    # input
    # -- pose: tuple (global position (x,y,z), euler angles (theta_x,theta_y,theta_z))
    # output
    # -- transformation: tuple/list (3x3 rotation matrix R, 3x1 translation vector t)
    assert isinstance(pose,tuple) or isinstance(pose,list)
    assert len(pose) == 2
    pos, angles = pose
    assert len(pos) == 3 and len(angles) == 3

    t = (np.array(pos)).reshape(3,)

    R = eulerAnglesToRotationMatrix(angles)

    return (R,t)

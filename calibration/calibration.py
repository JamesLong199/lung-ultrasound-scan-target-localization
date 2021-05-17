from CamCalib import CamCalib
import os
import numpy as np
import pickle

if __name__ == '__main__':
    myCam = CamCalib(w=8,h=6)
    # load images
    img_dir = '/home/james/projects/calibration/images'
    for i,img_name in enumerate(os.listdir(img_dir)):
        fname = img_dir + '/' + img_name
        myCam.add_image(fname)

    # calibrate camera
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = \
    myCam.run_calibration()

    np.set_printoptions(precision=3)
    # root mean square reprojection error in terms of pixels
    # between 0.1 and 1.0 for a good calibration
    print('retval:',retval)
    print('cameraMatrix:\n',cameraMatrix)
    print('distCoeffs:\n',distCoeffs)

    with open('cam_params.pickle','wb') as f:
        pickle.dump([cameraMatrix,distCoeffs],f)
        

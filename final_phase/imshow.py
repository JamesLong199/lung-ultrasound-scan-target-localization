import matplotlib.pyplot as plt
from subject_info import SUBJECT_NAME, SCAN_POSE

path1 = 'ViTPose_UR_data/' + SUBJECT_NAME + '/' + SCAN_POSE + '/color_images/cam_1.jpg'
path2 = 'ViTPose_UR_data/' + SUBJECT_NAME + '/' + SCAN_POSE + '/color_images/cam_2.jpg'
img_1 = plt.imread(path1)
img_2 = plt.imread(path2)
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img_1)
ax[1].imshow(img_2)
plt.show()
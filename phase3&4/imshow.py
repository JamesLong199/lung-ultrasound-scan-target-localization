import matplotlib.pyplot as plt

img_1 = plt.imread('OpenPose_UR_data/color_images/cam_0.jpg')
img_2 = plt.imread('OpenPose_UR_data/color_images/cam_1.jpg')
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img_1)
ax[1].imshow(img_2)
plt.show()
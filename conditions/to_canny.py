from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import cv2
def get_canny(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges
# image_path = '/media/sata4/Contextaware/image2condition_model/000000000000.jpg'
# image = cv2.imread(image_path)
# # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 100, 200)
# print("边缘像素数量:", np.count_nonzero(edges))
# cv2.imwrite("/media/sata4/Contextaware/image2condition_model/00canny_original.jpg", edges)
# edges = Image.fromarray(edges).convert("RGB")
# edges.save("/media/sata4/Contextaware/image2condition_model/00canny.jpg")
#             # return edges
# exit(0)
# lion_gray = np.dot(image[...,:3], [0.299, 0.587, 0.114])



# image_path='/media/sata4/Contextaware/image2condition_model/000000000002.jpg'
# Final_Image = get_canny(image_path)
# plt.imshow(Final_Image, cmap = plt.get_cmap('gray'))
# plt.savefig("canny.jpg")
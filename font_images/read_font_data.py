import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cv2



def hough_transform2(img):
    # img = cv2.imread('dave.jpg')
    # gray = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # edges = cv2.Canny(img, 50, 150, apertureSize=3)

    cv2.imwrite('test.jpg', img)

    line_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    lines = cv2.HoughLines(img, 10, np.pi / 180, 200)
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite('hough_line_test.jpg', line_image)




matrix = np.loadtxt('/home/student/workspace/testEncodings/kanji_output8/33203.dat', dtype=np.uint8, delimiter=',') * 255

hough_transform2(matrix)

# print(matrix)

# fig = plt.figure()
# ax1 = fig.add_subplot()
#
# ax1.imshow(matrix, interpolation='nearest', cmap=cm.Greys_r)
#
# plt.show()
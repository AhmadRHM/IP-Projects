import cv2
import numpy as np
from sklearn.cluster import MeanShift
from skimage.segmentation import slic, felzenszwalb
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
from skimage.util import img_as_float
from skimage import io


def main():
    img = cv2.imread("IMG_2805.jpg")
    # img = cv2.imread("Picture1.png")
    w = len(img[0])
    h = len(img)
    img = img[::2, ::2]
    number_of_segments = 500
    segments = slic(img, n_segments=number_of_segments, sigma=5)
    # segments = felzenszwalb(img_float, 100, sigma=5, min_size=50)
    number_of_segments = len(np.unique(segments))
    print("SLIC is done", number_of_segments)
    points = np.zeros((number_of_segments, 3))
    for segment in range(number_of_segments):
        segment_mask = segments == segment
        size_of_segment = segment_mask.sum()
        if size_of_segment != 0:
            for col in range(3):
                points[segment, col] = int((segment_mask * img[:, :, col]).sum() / size_of_segment)
    print("Points are made")
    ms = MeanShift(bandwidth=1)
    ms.fit_predict(points)
    print("Mean Shift done")
    cluster_centers = ms.cluster_centers_
    output = cluster_centers[ms.labels_[segments]]
    print(output)
    cv2.imshow("output", output.astype(np.uint8))
    cv2.imwrite("CartoonedImage.jpg", output.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

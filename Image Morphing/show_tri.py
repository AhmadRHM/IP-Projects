import numpy as np
import cv2
from scipy.spatial import Delaunay
from matplotlib import pyplot as plt


def main():
    first = cv2.imread("first.jpg")
    last = cv2.imread("last.jpg")
    first_points_file = open("first_points.txt", 'r')
    last_points_file = open("last_points.txt", 'r')

    n = int(next(first_points_file))
    first_points = np.zeros((n+5, 2))
    last_points = np.zeros_like(first_points)
    for i in range(n):
        x, y = [float(x) for x in next(first_points_file).split()]
        first_points[i] = (x, y)
        x, y = [float(x) for x in next(last_points_file).split()]
        last_points[i] = (x, y)

    w1, h1 = len(first[0]), len(first)
    w2, h2 = len(last[0]), len(last)
    w, h = max(w1, w2), max(h1, h2)
    first_points[n] = (0, 0)
    first_points[n + 1] = (w1 - 1, 0)
    first_points[n + 2] = (0, h1 - 1)
    first_points[n + 3] = (w1 - 1, h1 - 1)
    first_points[n + 4] = (w1 / 2, h1 - 1)
    last_points[n] = (0, 0)
    last_points[n + 1] = (w2 - 1, 0)
    last_points[n + 2] = (0, h2 - 1)
    last_points[n + 3] = (w2 - 1, h2 - 1)
    last_points[n + 4] = (w2 / 2, h2 - 1)

    tri = Delaunay(first_points)

    plt.imshow(first)
    plt.triplot(first_points[:, 0], first_points[:, 1], tri.simplices)
    plt.plot(first_points[:, 0], first_points[:, 1], 'o')
    plt.savefig("first_triangulated.jpg")
    plt.clf()
    plt.imshow(last)
    plt.triplot(last_points[:, 0], last_points[:, 1], tri.simplices)
    plt.plot(last_points[:, 0], last_points[:, 1], 'o')
    plt.savefig("last_triangulated.jpg")


if __name__ == '__main__':
    main()

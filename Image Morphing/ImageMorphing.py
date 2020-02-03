import numpy as np
import cv2
from scipy.spatial import Delaunay
import os

first, last = cv2.imread("first.jpg"), cv2.imread("last.jpg")


def first_points_getter(event, x, y, flags, param):
    global first
    file = open("first_points.txt", 'a')
    if event == cv2.EVENT_LBUTTONDOWN:
        file.write(str(x) + " " + str(y) + "\n")
        cv2.rectangle(first, (x-1, y-1), (x+1, y+1), (0, 0, 255), 2)
        cv2.imshow("first image - press s to save", first)
    file.close()


def last_points_getter(event, x, y, flags, param):
    global last
    file = open("last_points.txt", 'a')
    if event == cv2.EVENT_LBUTTONDOWN:
        file.write(str(x) + " " + str(y) + "\n")
        cv2.rectangle(last, (x - 1, y - 1), (x + 1, y + 1), (0, 0, 255), 2)
        cv2.imshow("last image - press s to save", last)
    file.close()


def main():
    first = cv2.imread("first.jpg").astype(np.float32)
    last = cv2.imread("last.jpg").astype(np.float32)

    has_point_file = input("Do you have selected point files?(y/n)")
    if has_point_file.split()[0] == 'n':
        first_image_name = "first image - press s to save"
        last_image_name = "last image - press s to save"
        open("first_points.txt", 'w').close()
        open("last_points.txt", 'w').close()
        cv2.namedWindow(first_image_name)
        cv2.namedWindow(last_image_name)
        cv2.setMouseCallback(first_image_name, first_points_getter)
        cv2.setMouseCallback(last_image_name, last_points_getter)
        cv2.imshow("first image - press s to save", first.astype(np.uint8))
        cv2.imshow("last image - press s to save", last.astype(np.uint8))
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                break
        cv2.destroyAllWindows()
        # calc n
        cnt = 0
        with open("first_points.txt", 'r') as f:
            for line in f:
                cnt += 1
            f.close()

        with open("first_points.txt", 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            line = str(cnt)
            f.write(line.rstrip('\r\n') + '\n' + content)
            f.close()

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
    first_points[n+1] = (w1-1, 0)
    first_points[n+2] = (0, h1-1)
    first_points[n+3] = (w1-1, h1-1)
    first_points[n+4] = (w1/2, h1-1)
    last_points[n] = (0, 0)
    last_points[n + 1] = (w2-1, 0)
    last_points[n + 2] = (0, h2-1)
    last_points[n + 3] = (w2 - 1, h2 - 1)
    last_points[n+4] = (w2/2, h2-1)

    tri = Delaunay(first_points)

    m = int(input("Enter the number of pics to be generated:").split()[0])

    path = os.path.join(os.getcwd(), "generated_pics")
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(m):
        t = (m - i - 1) / (m-1)
        cur = (t * first_points + (1-t) * last_points).astype(np.float32)
        output = np.zeros((h, w, 3))
        first_warped = np.zeros_like(output)
        last_warped = np.zeros_like(output)
        for j in range(len(tri.simplices)):
            inds = tri.simplices[j]
            mask_img = np.zeros_like(output)
            cv2.fillConvexPoly(mask_img, (cur[inds]).astype(np.int32), (255, 255, 255))
            mask = (mask_img == 255)
            # calc first warped
            warp_mat = cv2.getAffineTransform((first_points[inds]).astype(np.float32), cur[inds])
            first_warped[mask] = (cv2.warpAffine(first, warp_mat, (w, h)))[mask]
            # calc last warped
            warp_mat = cv2.getAffineTransform((last_points[inds]).astype(np.float32), cur[inds])
            last_warped[mask] = (cv2.warpAffine(last, warp_mat, (w, h)))[mask]
        output = t * first_warped + (1-t) * last_warped
        cv2.imwrite("generated_pics/output"+str(i+1)+".jpg", output.astype(np.uint8))
        print("the " + str(i+1) + "th image has been made")


if __name__ == '__main__':
    main()

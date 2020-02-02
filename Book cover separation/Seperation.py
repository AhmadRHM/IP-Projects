import numpy as np
import cv2


def distance(point1, point2):
    return np.sqrt(
        (point1[0] - point2[0]) * (point1[0] - point2[0]) + (point1[1] - point2[1]) * (point1[1] - point2[1]))


def main():
    img = cv2.imread("Books.jpg")

    book1 = [[607, 1105], [403, 802], [609, 663], [813, 966]]
    book2 = [[360, 753], [152, 724], [193, 429], [401, 458]]
    book3 = [[605, 402], [308, 294], [375, 108], [672, 216]]
    books = [book1, book2, book3]

    for i in range(len(books)):
        book = books[i]
        point1 = book[0]
        point2 = book[1]
        point3 = book[2]
        center = [int((point1[0] + point3[0]) / 2), int((point1[1] + point3[1]) / 2)]
        theta = np.rad2deg(np.arctan2(point2[1] - point1[1], point2[0] - point1[0]))
        l1 = distance(point1, point2)
        l2 = distance(point2, point3)
        if l1 < l2:
            tmp = l1
            l1 = l2
            l2 = tmp
            theta += 90
        cos = np.cos(np.deg2rad(90-theta))
        sin = np.sin(np.deg2rad(90-theta))
        mat1 = np.matrix([[cos, -sin, int(l2/2)],
                          [sin, cos, int(l1/2)],
                          [0, 0, 1]])
        mat2 = np.matrix([[1, 0, -center[0]],
                          [0, 1, -center[1]],
                          [0, 0, 1]])
        m = mat1 * mat2
        m = m.astype(list)
        matrix = np.zeros((2, 3))
        for k in range(2):
            for j in range(3):
                matrix[k, j] = m[k, j]
        print(matrix)
        output = cv2.warpAffine(img, matrix, (int(l2), int(l1)))
        cv2.imshow("output", output)
        cv2.imwrite("book" + str(i+1) + ".jpg", output)
        cv2.waitKey()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

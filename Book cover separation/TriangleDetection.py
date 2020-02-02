import numpy as np
import cv2

degree_err = 3
pixel_err = 5
go_further = 15


def draw_line(x1, y1, x2, y2, img):
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)
    if x1 == x2:
        for i in range(min(y1, y2), max(y1, y2)):
            img[i, x1] = 0
    elif y1 == y2:
        for i in range(min(x1, x2), max(x1, x2)):
            img[y1, i] = 0
    else:
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        for i in range(min(x1, x2), max(x1, x2) + 1):
            img[int(a * i + b), i] = 0
        for i in range(min(y1, y2), max(y1, y2) + 1):
            img[i, int((i - b) / a)] = 0


def get_coords(lower_right_point, theta, l1, l2):
    rad1 = np.deg2rad(theta)
    rad2 = np.deg2rad(90 - theta)
    vector_l1 = [int(-l1 * np.cos(rad1)), int(-l1 * np.sin(rad1))]
    vector_l2 = [int(l2 * np.cos(rad2)), int(-l2 * np.sin(rad2))]
    ret = list()
    ret.append(lower_right_point)
    ret.append([lower_right_point[0] + vector_l1[0], lower_right_point[1] + vector_l1[1]])
    ret.append([lower_right_point[0] + vector_l1[0] + vector_l2[0], lower_right_point[1] + vector_l1[1] + vector_l2[1]])
    ret.append([lower_right_point[0] + vector_l2[0], lower_right_point[1] + vector_l2[1]])
    return ret


def draw_rectangle1(lower_right_point, theta, l1, l2, img):
    nodes = get_coords(lower_right_point, theta, l1, l2)
    print("rectangle points: ")
    for i in range(4):
        print(nodes[i][0], nodes[i][1])
        if nodes[i][0] < 0 or nodes[i][0] > len(img[0]) or nodes[i][1] < 0 or nodes[i][1] > len(img):
            return
    for channel in range(3):
        for i in range(4):
            draw_line(nodes[i][0], nodes[i][1], nodes[(i + 1) % 4][0], nodes[(i + 1) % 4][1], img[:, :, channel])


def draw_rectangle2(lower_point, upper_point, theta, img):
    dx = lower_point[0] - upper_point[0]
    dy = lower_point[1] - upper_point[1]
    sin = np.sin(np.deg2rad(theta))
    cos = np.cos(np.deg2rad(theta))
    l2 = cos * dy - sin * dx
    l1 = (dy - cos * l2) / sin
    draw_rectangle1(lower_point, theta, l1, l2, img)


def distance(point1, point2):
    return np.sqrt(
        (point1[0] - point2[0]) * (point1[0] - point2[0]) + (point1[1] - point2[1]) * (point1[1] - point2[1]))


def main():
    img = cv2.imread("EdgePoints.jpg", cv2.IMREAD_GRAYSCALE)
    deg = cv2.imread("GradientDirection.jpg", cv2.IMREAD_GRAYSCALE)

    h = len(img)
    w = len(img[0])

    edges = list()

    lower_right = np.zeros((h, w))
    for y in range(h - 1, -1, -1):
        for x in range(w):
            if img[y, x] == 255:
                edges.append((x, y))
                # vote to lower right point of the triangle
                gradient_degree = deg[y, x]
                gradient_radian = np.deg2rad(gradient_degree)
                rho = x * np.cos(gradient_radian) + y * np.sin(gradient_radian)
                for dy in range(go_further):
                    newy = y + dy
                    if gradient_degree == 90:
                        newx = x
                    else:
                        newx = int((rho - newy * np.sin(gradient_radian)) / np.cos(gradient_radian))
                    if newy < h and 0 <= newx < w:
                        lower_right[newy, newx] += 1

    lower_right_threshold = lower_right.max() - 5
    lower_right_min_dist = h / 3
    lower_right_points = list()
    for y in range(h - 1, -1, -1):
        for x in range(w):
            point = [x, y]
            if lower_right[point[1], point[0]] >= lower_right_threshold:
                mi = np.inf
                for chosen_point in lower_right_points:
                    mi = min(mi, np.sqrt(
                        (point[0] - chosen_point[0]) * (point[0] - chosen_point[0]) + (point[1] - chosen_point[1]) * (
                                point[1] - chosen_point[1])))
                if mi >= lower_right_min_dist:
                    lower_right_points.append(point)

    # showing the result of finding lower right points of rectangles
    print(lower_right_points)
    # output = np.zeros((h, w, 3))
    # output[:, :, 0] = lower_right
    # output[:, :, 1] = (lower_right >= lower_right_threshold) * np.ones((h, w)) * 255
    # output[:, :, 2] = (lower_right >= lower_right_threshold) * np.ones((h, w)) * 255
    # cv2.imshow("lower_right", cv2.resize(output, (int(w / 2), int(h / 2))))
    # ---------------------------------------lower right point found-------------------------------------------------
    upper_left = np.zeros((h, w))
    for edge_point in edges:
        x = edge_point[0]
        y = edge_point[1]
        gradient_degree = deg[y, x]
        gradient_radian = np.deg2rad(gradient_degree)
        rho = x * np.cos(gradient_radian) + y * np.sin(gradient_radian)
        for dy in range(go_further):
            newy = y - dy
            if gradient_degree == 90:
                newx = x
            else:
                newx = int((rho - newy * np.sin(gradient_radian)) / np.cos(gradient_radian))
            if 0 <= newy < h and 0 <= newx < w:
                upper_left[newy, newx] += 1

    upper_left_threshold = upper_left.max() - 5
    upper_left_min_dist = h / 3 - 60
    upper_left_points = list()
    for y in range(h):
        for x in range(w):
            if len(upper_left_points) >= len(lower_right_points):
                break
            point = [x, y]
            if upper_left[point[1], point[0]] >= upper_left_threshold:
                mi = np.inf
                for chosen_point in upper_left_points:
                    mi = min(mi, np.sqrt(
                        (point[0] - chosen_point[0]) * (point[0] - chosen_point[0]) + (point[1] - chosen_point[1]) * (
                                point[1] - chosen_point[1])))
                if mi >= upper_left_min_dist:
                    upper_left_points.append(point)
    # showing the result of finding upper left points of rectangles
    print(upper_left_points)
    output = np.zeros((h, w, 3))
    output[:, :, 0] = upper_left
    output[:, :, 1] = (upper_left >= upper_left_threshold) * np.ones((h, w)) * 255
    output[:, :, 2] = (upper_left >= upper_left_threshold) * np.ones((h, w)) * 255
    cv2.imshow("upper left", cv2.resize(output, (int(w / 2), int(h / 2))))
    # -----------------------------------------upper left point found---------------------------------------------
    theta_vote_lower_right = np.zeros((len(lower_right_points), 180))
    theta_vote_upper_left = np.zeros((len(lower_right_points), 180))

    for edge_point in edges:
        for i in range(len(lower_right_points)):
            lower_right_point = lower_right_points[i]
            upper_left_point = upper_left_points[i]
            gradient_degree = deg[edge_point[1], edge_point[0]]
            gradient_radian = np.deg2rad(gradient_degree)
            rho1 = edge_point[0] * np.cos(gradient_radian) + edge_point[1] * np.sin(gradient_radian)
            rho2 = lower_right_point[0] * np.cos(gradient_radian) + lower_right_point[1] * np.sin(gradient_radian)
            rho3 = upper_left_point[0] * np.cos(gradient_radian) + upper_left_point[1] * np.sin(gradient_radian)
            if np.abs(rho1 - rho2) <= 5:
                if gradient_degree >= 90:
                    gradient_degree -= 90
                theta_vote_lower_right[(i, gradient_degree)] += 1
            if np.abs(rho1 - rho3) <= 5:
                if gradient_degree >= 90:
                    gradient_degree -= 90
                theta_vote_upper_left[(i, gradient_degree)] += 1

    theta_of_lower_point = []
    theta_of_upper_point = []
    for i in range(len(lower_right_points)):
        theta = theta_vote_lower_right[i].argmax()
        theta_of_lower_point.append(theta)
        theta = theta_vote_upper_left[i].argmax()
        theta_of_upper_point.append(theta)
    # ----------------------------------best theta of each upper and lower point found--------------------------------
    output = cv2.imread("Books.jpg")

    for i in range(len(lower_right_points)):
        lower_right_point = lower_right_points[i]
        chosen_upper_left_point = upper_left_points[0]
        for j in range(len(upper_left_points)):
            upper_left_point = upper_left_points[j]
            if np.abs(theta_of_upper_point[j] - theta_of_lower_point[i]) <= 5 and distance(chosen_upper_left_point,
                                                                                           lower_right_point) >= distance(
                upper_left_point, lower_right_point) and upper_left_point[1] <= lower_right_point[1]:
                chosen_upper_left_point = upper_left_point
        draw_rectangle2(lower_right_point, chosen_upper_left_point, theta_of_lower_point[i], output)
    cv2.imshow("rectangles", cv2.resize(output, (int(w / 2), int(h / 2))))
    cv2.imwrite("FoundRectangles.jpg", output)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

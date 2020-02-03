import numpy as np
import cv2
from matplotlib import pyplot as plt
import os


def get_abs_gradient(img):
    sigma = 3
    smoothed = cv2.GaussianBlur(img, (6 * sigma + 1, 6 * sigma + 1), sigma)
    gradient_x = smoothed[:-1, :] - smoothed[1:, :]
    gradient_y = smoothed[:, :-1] - smoothed[:, 1:]

    ret = (gradient_x * gradient_x)[:, :-1] + (gradient_y * gradient_y)[:-1, :]
    none_zero_mask = ret > 20
    print(ret.sum())
    ret = ret * none_zero_mask
    print(ret.sum())
    return ret


def calc_dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_coord(x, y, state, window_size):
    x += int(state / window_size) - 1
    y += int(state % window_size) - 1
    return x, y


def is_in(x, y, shape):
    return 0 <= x < shape[0] and 0 <= y < shape[1]


def get_mean_d(points):
    n = len(points)
    mean_d = 0
    for i in range(n):
        next = (i + 1) % n
        mean_d += calc_dist(points[i, 0], points[i, 1], points[next, 0], points[next, 1])
    mean_d /= n
    return mean_d


def calc_energy(points, abs_gradient, mean_d):
    internal_energy = calc_internal_energy(points, mean_d)
    external_energy = calc_external_energy(points, abs_gradient)
    landa = 400
    return internal_energy + external_energy * landa


def calc_internal_energy_of_point(points, mean_d, i):
    n = len(points)
    alpha = 1
    betha = 50
    ret = 0
    im1 = (i - 1 + n) % n
    ip1 = (i + 1) % n
    ret += alpha * ((mean_d - calc_dist(points[i, 0], points[i, 1], points[ip1, 0], points[ip1, 0])) ** 2)
    ret += betha * (calc_dist(points[im1, 0] + points[ip1, 0], points[im1, 1] + points[ip1, 1], 2 * points[i, 0],
                              2 * points[i, 1])) ** 2
    return ret


def calc_internal_energy(points, mean_d):
    n = len(points)
    ret = 0
    for i in range(n):
        ret += calc_internal_energy_of_point(points, mean_d, i)
    return ret


def calc_external_energy(points, abs_gradient):
    n = len(points)
    ret = 0
    for i in range(n):
        ret -= abs_gradient[int(points[i, 0]), int(points[i, 1])]
    return ret


def init_points():
    s = np.linspace(0, 2 * np.pi, 300)
    x, y = 400, 550
    radius1, radius2 = 250, 400
    r = x + radius1 * np.sin(s)
    c = y + radius2 * np.cos(s)
    nim_beyzi = np.array([r, c]).T
    return nim_beyzi


rnd = np.random.permutation(300)
def iterate_with_greedy(points, abs_gradient):
    n = len(points)
    inf = 1000_000_000_000
    window_size = 3
    number_of_states = window_size ** 2
    mean_d = get_mean_d(points)
    mean_d *= 0.95
    for i in rnd:
        best_energy = inf
        best_state = (number_of_states + 1) / 2
        for state in range(number_of_states):
            newx, newy = get_coord(points[i, 0], points[i, 1], state, window_size)
            if is_in(newx, newy, abs_gradient.shape):
                oldx, oldy = points[i, 0], points[i, 1]
                points[i, 0], points[i, 1] = newx, newy
                landa = 400
                alpha = 1
                betha = 50
                delta_external_energy = -abs_gradient[int(points[i, 0]), int(points[i, 1])]
                ip1 = (i + 1) % n
                im1 = (i - 1 + n) % n
                # elastic energy
                delta_internal_energy = alpha * ((mean_d - calc_dist(points[i, 0], points[i, 1], points[ip1, 0], points[ip1, 0])) ** 2)
                delta_internal_energy += alpha * ((mean_d - calc_dist(points[i, 0], points[i, 1], points[im1, 0], points[im1, 0])) ** 2)
                delta_internal_energy += betha * ((
                    calc_dist(points[im1, 0] + points[ip1, 0], points[im1, 1] + points[ip1, 1], 2 * points[i, 0],
                              2 * points[i, 1])) ** 2)
                delta_energy = delta_internal_energy + landa * delta_external_energy
                points[i, 0], points[i, 1] = oldx, oldy
                if delta_energy < best_energy:
                    best_energy = delta_energy
                    best_state = state
        best_x, best_y = get_coord(points[i, 0], points[i, 1], best_state, window_size)
        if is_in(best_x, best_y, abs_gradient.shape):
            points[i, 0], points[i, 1] = best_x, best_y


def main():
    img = cv2.imread("tasbih.jpg", cv2.IMREAD_GRAYSCALE)

    abs_gradient = get_abs_gradient(img)

    img = cv2.imread("tasbih.jpg")

    points = init_points()
   
    max_iteration = 1000
    iteration_number = 0
    delta_energy = 20

    path = os.path.join(os.getcwd(), "GeneratedImages")
    if not os.path.exists(path):
        os.mkdir(path)
    while iteration_number < max_iteration:
        mean_d = get_mean_d(points)
        e_bef = calc_energy(points, abs_gradient, mean_d)
        iterate_with_greedy(points, abs_gradient)
        e_aft = calc_energy(points, abs_gradient, mean_d)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)
        ax.plot(points[:, 1], points[:, 0], 'ro')
        ax.set_xticks([]), ax.set_yticks([])
        plt.savefig("GeneratedImages/img" + str(iteration_number) + ".jpg")
        plt.close()
        iteration_number += 1
        print(e_bef, e_aft)
        if abs(e_aft - e_bef) < delta_energy:
            break
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    ax.plot(points[:, 1], points[:, 0], 'ro')
    ax.set_xticks([]), ax.set_yticks([])
    plt.savefig("FinalImage.jpg")
    plt.close()


if __name__ == '__main__':
    main()

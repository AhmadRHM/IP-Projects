import numpy as np
import cv2
from random import randint
from skimage import color

patch_size = 20
overlap = 5
alpha = 1000_000  # constant of importance of being like target than being like sample


def intensity_difference(sample, patch):  # patch is from target pic
    w, h = len(sample[0]), len(sample)
    ssd = np.zeros((h - patch_size, w - patch_size))
    sample_gray = color.rgb2gray(sample)
    for x in range(patch_size):
        for y in range(patch_size):
            sq = np.square(sample_gray[y:h - patch_size + y, x:w - patch_size + x] - patch[y, x])
            ssd += sq
    return ssd


def find_best_top(data, sample):
    w, h = len(sample[0]), len(sample)
    ssd = np.zeros((h - patch_size, w - patch_size))
    for x in range(patch_size):
        for y in range(overlap):
            sq = np.square(sample[y:h - patch_size + y, x:w - patch_size + x] - data[y, x])
            ssd += sq[:, :, 0] + sq[:, :, 1] + sq[:, :, 2]
    return ssd


def find_best_left(data, sample):
    w, h = len(sample[0]), len(sample)
    ssd = np.zeros((h - patch_size, w - patch_size))
    for x in range(overlap):
        for y in range(patch_size):
            sq = np.square(sample[y:h - patch_size + y, x:w - patch_size + x] - data[y, x])
            ssd += sq[:, :, 0] + sq[:, :, 1] + sq[:, :, 2]
    return ssd


def find_best_top_left(data, sample):
    w, h = len(sample[0]), len(sample)
    ssd = find_best_top(data, sample)
    for x in range(overlap):
        for y in range(overlap, patch_size):
            sq = np.square(sample[y:h - patch_size + y, x:w - patch_size + x] - data[y, x])
            ssd += sq[:, :, 0] + sq[:, :, 1] + sq[:, :, 2]
    return ssd


def calc_best_cut_line(cost):
    dp = np.ones((patch_size, overlap)) * 1000_000_000_000_000_000_000
    par = np.zeros((patch_size, overlap))
    # initializing
    for i in range(overlap):
        dp[0, i] = cost[i, 0]
    # calc dp
    for x in range(1, patch_size):
        for state in range(overlap):
            dp[x, state] = dp[x - 1, state]
            par[x, state] = state
            if state != 0 and dp[x - 1, state - 1] < dp[x, state]:
                dp[x, state], par[x, state] = dp[x - 1, state - 1], state - 1
            if state + 1 != overlap and dp[x - 1, state + 1] < dp[x, state]:
                dp[x, state], par[x, state] = dp[x - 1, state + 1], state + 1
            dp[x, state] += cost[state, x]
    min_state_of_last = 0
    for i in range(overlap):
        if dp[patch_size-1, i] < dp[patch_size-1, min_state_of_last]:
            min_state_of_last = i
    return int(min_state_of_last), par


def mix_patch_top(patch, output, x, y):
    bef_data = output[y:y+overlap, x:x+patch_size]
    cost = np.abs(bef_data - patch[:overlap, :])
    cost = cost[:, :, 0] + cost[:, :, 1] + cost[:, :, 2]
    cur_state, par = calc_best_cut_line(cost)
    for cur in range(patch_size-1, -1, -1):
        output[y+cur_state:y+overlap, x+cur] = patch[cur_state:overlap, cur]
        if cur == 0:
            break
        cur_state = int(par[cur, cur_state])
    output[y + overlap:y + patch_size, x:x + patch_size] = patch[overlap:, :]


def mix_patch_left(patch, output, x, y):
    bef_data = output[y:y + patch_size, x:x + overlap]
    cost = np.abs(bef_data - patch[:, :overlap])
    cost = cost[:, :, 0] + cost[:, :, 1] + cost[:, :, 2]
    cost = cost.transpose()
    cur_state, par = calc_best_cut_line(cost)
    for cur in range(patch_size-1, -1, -1):
        output[y+cur, x+cur_state:x+overlap] = patch[cur, cur_state:overlap]
        if cur == 0:
            break
        cur_state = int(par[cur, cur_state])
    output[y:y + patch_size, x + overlap:x + patch_size] = patch[:, overlap:]


def mix_patch_top_left(patch, output, x, y):
    # left data
    patch = np.copy(patch)
    bef_data_left = output[y:y + patch_size, x:x + overlap]
    cost_left = np.abs(bef_data_left - patch[:, :overlap])
    cost_left = cost_left[:, :, 0] + cost_left[:, :, 1] + cost_left[:, :, 2]
    cost_left = cost_left.transpose()
    cur_state, par = calc_best_cut_line(cost_left)
    for cur in range(patch_size-1, -1, -1):
        patch[cur, :cur_state] = output[y+cur, x:x+cur_state]
        if cur == 0:
            break
        cur_state = int(par[cur, cur_state])
    # top data
    bef_data_top = output[y:y + overlap, x:x + patch_size]
    cost_top = np.abs(bef_data_top - patch[:overlap, :])
    cost_top = cost_top[:, :, 0] + cost_top[:, :, 1] + cost_top[:, :, 2]
    cur_state, par = calc_best_cut_line(cost_top)
    for cur in range(patch_size-1, -1, -1):
        patch[:cur_state, cur] = output[y:y+cur_state, x+cur]
        if cur == 0:
            break
        cur_state = int(par[cur, cur_state])
    output[y:y + patch_size, x:x + patch_size] = patch


def main():
    sample = cv2.imread("mat.jpg")
    sample_gray = color.rgb2gray(sample)
    target = cv2.imread("mali.jpg", cv2.IMREAD_GRAYSCALE)
    target = target.astype(np.float64)
    l1, r1 = target.min(), target.max()
    l2, r2 = sample_gray.min(), sample_gray.max()
    d1, d2 = r1 - l1, r2 - l2
    target *= d2 / d1
    l1 *= d2 / d1
    target += l2 - l1
    sample_w, sample_h = len(sample[0]), len(sample)

    w, h = len(target[0]), len(target)
    print(w, h, w - patch_size, h - patch_size)
    output = np.zeros((h, w, 3))
    # initialization of output
    rand_x, rand_y = randint(0, sample_w - patch_size - 1), randint(0, sample_h - patch_size - 1)
    output[:patch_size, :patch_size] = sample[rand_y:rand_y + patch_size, rand_x:rand_x + patch_size]
    for x in range(0, w - patch_size + 1, patch_size - overlap):
        for y in range(0, h - patch_size + 1, patch_size - overlap):
            print("filling ", x, y)
            data = output[y:y + patch_size, x:x + patch_size]
            if x == y == 0:
                continue
            if x == 0:
                ssd = find_best_top(data, sample)
            elif y == 0:
                ssd = find_best_left(data, sample)
            else:
                ssd = find_best_top_left(data, sample)
            ssd += alpha * intensity_difference(sample, target[y:y+patch_size, x:x+patch_size])
            ind = randint(0, 10)
            ssd_of_selected_patch = np.partition(np.reshape(ssd, len(ssd) * len(ssd[0])), ind)[ind]
            itemindex = np.where(ssd == ssd_of_selected_patch)
            found_x, found_y = itemindex[1][0], itemindex[0][0]
            patch = sample[found_y:found_y + patch_size, found_x:found_x + patch_size]
            if x == 0:
                mix_patch_top(patch, output, int(x), int(y))
            elif y == 0:
                mix_patch_left(patch, output, int(x), int(y))
            else:
                mix_patch_top_left(patch, output, int(x), int(y))
    cv2.imshow("output", output.astype(np.uint8))
    cv2.imshow("target", target)
    cv2.imwrite("TextureTransferred.jpg", output.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

import numpy as np
from itertools import cycle, islice
import cv2
import math


COCO_JOINT_NAMES = [
    'nose',
    'left eye',
    'right eye',
    'left ear',
    'right ear',
    'left shoulder',
    'right shoulder',
    'left elbow',
    'right elbow',
    'left wrist',
    'right wrist',
    'left hip',
    'right hip',
    'left knee',
    'right knee',
    'left ankle',
    'right ankle',
    'neck',
    'middle shoulders',
    'middle hips',
]

coco2openpose = np.array([
    [0, 0],
    [1, 14],
    [2, 15],
    [3, 16],
    [4, 17],
    [5, 2],
    [6, 5],
    [7, 3],
    [8, 6],
    [9, 4],
    [10, 7],
    [11, 8],
    [12, 11],
    [13, 9],
    [14, 12],
    [15, 10],
    [16, 13],
    [17, 1],
])

casia_idx2variation = {
    0: 'bg-0',
    1: 'bg-18',
    2: 'bg-36',
    3: 'bg-54',
    4: 'bg-72',
    5: 'bg-90',
    6: 'bg-108',
    7: 'bg-126',
    8: 'bg-144',
    9: 'bg-162',
    10: 'bg-180',
    11: 'cl-0',
    12: 'cl-18',
    13: 'cl-36',
    14: 'cl-54',
    15: 'cl-72',
    16: 'cl-90',
    17: 'cl-108',
    18: 'cl-126',
    19: 'cl-144',
    20: 'cl-162',
    21: 'cl-180',
    22: 'nm-0',
    23: 'nm-18',
    24: 'nm-36',
    25: 'nm-54',
    26: 'nm-72',
    27: 'nm-90',
    28: 'nm-108',
    29: 'nm-126',
    30: 'nm-144',
    31: 'nm-162',
    32: 'nm-180'
}
casia_variation2idx = {name: idx for idx, name in casia_idx2variation.items()}

fvg_idx2variation = {0: 'nm', 1: 'ws', 2: 'cb', 3: 'cl', 4: 'cbg'}
fvg_variation2idx = {name: idx for idx, name in fvg_idx2variation.items()}

idx2variation = {
    'casia': casia_idx2variation,
    'fvg': fvg_idx2variation,
}

variation2idx: {
    'casia': casia_variation2idx,
    'fvg': fvg_variation2idx,
}

openpose2coco = np.array([coco2openpose[:, 1], coco2openpose[:, 0]]).T

coco_pairs = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (17, 11), (17, 12),  # Body
    (11, 13), (12, 14), (13, 15), (14, 16),
    (11, 17), (12, 17),
    (5, 18), (6, 18), (18, 17)
]


def compute_colors(linecolor = None, pcolor = None, differentiate_left_right = False):
    p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
               (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
               (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
    line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                  (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                  (77, 222, 255), (255, 156, 127),
                  (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]

    if linecolor is not None:
        line_color = [linecolor for _ in range(len(coco_pairs))]
    else:
        line_color = list(islice(cycle(line_color), len(coco_pairs)))

    if pcolor is not None:
        p_color = [pcolor for _ in range(19)]
    else:
        p_color.append((0, 255, 255))

    if not differentiate_left_right:
        return line_color, p_color

    left_indices = [1, 3, 5, 7, 9, 11, 13, 15]
    right_indices = [2, 4, 6, 8, 10, 12, 14, 16]
    neutral_indices = [0, 17, 18]

    left_color = (66, 135, 245)
    right_color = (245, 158, 66)
    neutral_color = (66, 245, 75)

    for idx in left_indices:
        p_color[idx] = left_color

    for idx in right_indices:
        p_color[idx] = right_color

    for idx in neutral_indices:
        p_color[idx] = neutral_color

    for i, (joint_1, joint_2) in enumerate(coco_pairs):
        if joint_1 in left_indices or joint_2 in left_indices:
            line_color[i] = left_color
        elif joint_1 in right_indices or joint_2 in right_indices:
            line_color[i] = right_color
        else:
            line_color[i] = neutral_color

    return line_color, p_color


def draw_pose(frame, poses, pcolor = None, linecolor = None, differentiate_left_right = False, cs = 1):
    line_color, p_color = compute_colors(pcolor, linecolor, differentiate_left_right)

    img = frame.copy()
    height, width = img.shape[:2]
    img = cv2.resize(img, (width // 2, height // 2))

    middle_hips = (poses[:, 12, :] + poses[:, 11, :]) / 2
    middle_hips = middle_hips.reshape((poses.shape[0], 1, 3))
    poses = np.hstack((poses, middle_hips))

    if poses.shape[1] == 18:
        middle_shoulders = (poses[:, 5, :] + poses[:, 6, :]) / 2
        middle_shoulders = middle_shoulders.reshape((poses.shape[0], 1, 3))
        poses = np.hstack((poses, middle_shoulders))

    for human in poses:
        part_line = {}
        for n in range(human.shape[0]):
            cor_x, cor_y = int(human[n, 0]), int(human[n, 1])
            part_line[n] = (int(cor_x / 2), int(cor_y / 2))
            if human[n, 2] < 0.3:
                cv2.circle(img, (int(cor_x / 2), int(cor_y / 2)), cs, (255, 0, 0), -1)
            else:
                cv2.circle(img, (int(cor_x / 2), int(cor_y / 2)), cs, p_color[n], -1)

        for i, (start_p, end_p) in enumerate(coco_pairs):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                bg = img.copy()

                X = (start_xy[0], end_xy[0])
                Y = (start_xy[1], end_xy[1])
                mX = np.mean(X)
                mY = np.mean(Y)
                length = ((Y[0] - Y[1]) ** 2 + (X[0] - X[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(Y[0] - Y[1], X[0] - X[1]))

                stickwidth = (human[start_p, 2] + human[end_p, 2])
                stickwidth = 1

                transparency = max(0, min(1, 0.5 * (human[start_p, 2] + human[end_p, 2])))
                transparency = 1.

                if human[start_p, 2] == 0 or human[end_p, 2] == 0:
                    continue

                polygon = cv2.ellipse2Poly((int(mX), int(mY)), (int(length / 2), int(stickwidth)), int(angle), 0, 360, 1)
                cv2.fillConvexPoly(bg, polygon, line_color[i])
                img = cv2.addWeighted(bg, transparency, img, 1 - transparency, 0)

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    return img

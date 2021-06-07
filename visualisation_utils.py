import numpy as np
import cv2

KEYPOINT_DICT = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


BODY_LINES = [
    [4, 2],
    [2, 0],
    [0, 1],
    [1, 3],
    [10, 8],
    [8, 6],
    [6, 5],
    [5, 7],
    [7, 9],
    [6, 12],
    [12, 11],
    [11, 5],
    [12, 14],
    [14, 16],
    [11, 13],
    [13, 15],
]


def visualise(frame, coords, scores, score_thresh):
    lines = [
        np.array([coords[point] for point in line], dtype=np.int32)
        for line in BODY_LINES
        if scores[line[0]] > score_thresh and scores[line[1]] > score_thresh
    ]
    cv2.polylines(frame, lines, False, (255, 180, 90), 2, cv2.LINE_AA)

    for i, x_y in enumerate(coords):
        if scores[i] > score_thresh:
            if i % 2 == 1:
                color = (0, 255, 0)
            elif i == 0:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            cv2.circle(frame, (int(x_y[0]), int(x_y[1])), 4, color, -11)
    return frame

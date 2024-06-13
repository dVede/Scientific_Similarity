from math import sqrt

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance

def get_ele_size(box):
    return box[2], box[3]

def get_ele_center(box):
    return box[0] + box[2]/2.0, box[1] + box[3]/2.0

def get_shape_diff(box_a, box_b):
    width_a, height_a = get_ele_size(box_a)
    width_b, height_b = get_ele_size(box_b)
    return abs(width_b - width_a) + abs(height_b - height_a)

def get_pos_diff(box_a, box_b):
    center_a = get_ele_center(box_a)
    center_b = get_ele_center(box_b)
    return distance.euclidean(center_a, center_b)

def get_area_factor(box_a, box_b):
    width_a, height_a = get_ele_size(box_a)
    width_b, height_b = get_ele_size(box_b)
    return sqrt(min(width_a * height_a, width_b * height_b))


def get_ele_sim(box_a, label_a, box_b, label_b):
    if label_a != label_b:
        return 0
    pos_diff = get_pos_diff(box_a, box_b)
    shape_diff = get_shape_diff(box_a, box_b)
    area_factor = get_area_factor(box_a, box_b)
    return area_factor * (pow(2, -pos_diff - 2 * shape_diff))

def get_layout_sim(boxes_a, labels_a, boxes_b, labels_b):
    ele_sim = [
        [get_ele_sim(box_a, label_a, box_b, label_b) for box_b, label_b in zip(boxes_b, labels_b)]
        for box_a, label_a in zip(boxes_a, labels_a)
    ]
    cost_matrix = np.array(ele_sim)
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

    return cost_matrix[row_ind, col_ind].sum()
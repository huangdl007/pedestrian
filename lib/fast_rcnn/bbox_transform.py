# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
#import tracback

def bbox_transform(ex_rois, gt_rois):
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)
    # if gt_heights[0] < 0 or gt_widths[0] < 0:
    #     print "gt_widths:", gt_widths
    #     print "gt_heights:", gt_heights
    #     print "ex_widths:", ex_widths
    #     print "ex_heights:", ex_heights
    #     print "gt_rois:", gt_rois
    #     print "ex_rois:", ex_rois
    #     print fsdgf
    
    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)
    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def landmark_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, 6), dtype=deltas.dtype)
    
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0

    #assert all landmarks inside the boxes
    deltas = np.maximum(np.minimum(deltas, 1.0), 0.0)
    #assert deltas.all() <= 1.0, 'all landmarks must be inside the boxes'
    # inds = np.where(deltas<0.0)[0]
    # if len(inds) > 0:
    #     print deltas
    #     exit(1)
    dhx = deltas[:, 0::6]
    dhy = deltas[:, 1::6]
    dlx = deltas[:, 2::6]
    dly = deltas[:, 3::6]
    drx = deltas[:, 4::6]
    dry = deltas[:, 5::6]

    pred_landmarks = np.zeros((deltas.shape[0], 6), dtype=deltas.dtype)
    pred_landmarks[:, 0::6] = dhx * widths[:, np.newaxis]
    pred_landmarks[:, 1::6] = dhy * heights[:, np.newaxis]
    pred_landmarks[:, 2::6] = dlx * widths[:, np.newaxis]
    pred_landmarks[:, 3::6] = dly * heights[:, np.newaxis]
    pred_landmarks[:, 4::6] = drx * widths[:, np.newaxis]
    pred_landmarks[:, 5::6] = dry * heights[:, np.newaxis]

    return pred_landmarks

def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes

def clip_landmarks(landmarks, boxes):
    """
    Clip landmarks inside boxes boundaries.
    """
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    # hx <= width
    landmarks[:, 0::6] = np.minimum(landmarks[:, 0::6], widths[:, np.newaxis])
    # hy <= height
    landmarks[:, 1::6] = np.minimum(landmarks[:, 1::6], heights[:, np.newaxis])
    # lx <= width
    landmarks[:, 2::6] = np.minimum(landmarks[:, 2::6], widths[:, np.newaxis])
    # ly <= height
    landmarks[:, 3::6] = np.minimum(landmarks[:, 3::6], heights[:, np.newaxis])
    # rx <= width
    landmarks[:, 4::6] = np.minimum(landmarks[:, 4::6], widths[:, np.newaxis])
    # ry <= height
    landmarks[:, 5::6] = np.minimum(landmarks[:, 5::6], heights[:, np.newaxis])

    return landmarks
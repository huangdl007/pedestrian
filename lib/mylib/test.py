from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
from utils.cython_bbox import bbox_overlaps
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
import cPickle
from utils.blob import im_list_to_blob
import os

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def im_detect(net, im, boxes=None):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals or None (for RPN)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    if cfg.TEST.HAS_RPN:
        net.blobs['im_info'].reshape(*(blobs['im_info'].shape))
    else:
        net.blobs['rois'].reshape(*(blobs['rois'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    if cfg.TEST.HAS_RPN:
        forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    else:
        forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        rois = net.blobs['rois'].data.copy()
        #print rois.shape
        #exit(1)
        # unscale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]

    return boxes

def accuracy_detect(net, im, boxes=None, labels=None):
    blobs, im_scales = _get_blobs(im, boxes)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['rois'].reshape(*(blobs['rois'].shape))
    net.blobs['labels'].reshape(*(labels.shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    forward_kwargs['rois'] = blobs['rois'].astype(np.float32, copy=False)
    forward_kwargs['labels'] = labels.astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)

    assert len(im_scales) == 1, "Only single-image batch implemented"
    #print net.blobs.items()
    accuracy = net.blobs['accuracy'].data.copy()
    probs = net.blobs['rpn_cascade_cls_prob'].data.copy()
    probs.reshape((300, 2))
    return accuracy, probs

def test_rpn(net, imdb, max_per_image=100, thresh=0.5, vis=False, wrt=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    if not cfg.TEST.HAS_RPN:
        roidb = imdb.roidb

    #result image output dir
    if wrt:
        image_output_dir = os.path.join(output_dir, 'result_imgs')
        if not os.path.isdir(image_output_dir):
            os.mkdir(image_output_dir)

    for i in xrange(num_images):
        # filter out any ground truth boxes
        if cfg.TEST.HAS_RPN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select those the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = roidb[i]['boxes'][roidb[i]['gt_classes'] == 0]

        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        rois_proposals = im_detect(net, im, box_proposals)
        
        
        #write result image
        if wrt:
            import copy
            result_im = copy.copy(im)
            savename = os.path.join(image_output_dir, os.path.basename(imdb.image_path_at(i)))
            for j in range(len(rois_proposals)):
                box = rois_proposals[j]
                cv2.rectangle(result_im, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))
                #cv2.putText(result_im, '{}: {:.3f}'.format(imdb.classes[j], score), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
            cv2.imwrite(savename, result_im)
        
        _t['im_detect'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s, {:d} roi proposals' \
              .format(i + 1, num_images, _t['im_detect'].average_time, len(rois_proposals))
    print 'RPN Testing Finished!'

def get_rois_labels(rois, gt_boxes):
    labels = np.zeros((len(rois), 1), dtype=np.float32)

    gt_overlaps = bbox_overlaps(rois.astype(np.float),
                                            gt_boxes.astype(np.float))
    max_overlaps = gt_overlaps.max(axis=1)
    I = np.where(max_overlaps >= 0.5)[0]
    labels[I, :] = 1

    return labels

def test_rpn_cascade_accuracy(net, imdb, max_per_image=100, thresh=0.5, vis=False, wrt=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    accuracys = [[] for _ in xrange(num_images)]
    label_nums = [[] for _ in xrange(num_images)]

    # timers
    _t = {'accuracy_detect' : Timer(), 'misc' : Timer()}

    output_dir = get_output_dir(imdb, net)
    #result image output dir
    if wrt:
        image_output_dir = os.path.join(output_dir, 'result_imgs')
        if not os.path.isdir(image_output_dir):
            os.mkdir(image_output_dir)

    roidb = imdb.roidb
    gt_roidb = imdb.gt_roidb()

    right_num = 0.0
    total = 0.0
    for i in xrange(num_images):
        rois = roidb[i]['boxes']
        labels = get_rois_labels(rois, gt_roidb[i]['boxes'])

        _t['accuracy_detect'].tic()
        im = cv2.imread(imdb.image_path_at(i))
        accuracy, probs = accuracy_detect(net, im, rois, labels)
        #write result image
        if wrt:
            import copy
            result_im = copy.copy(im)
            savename = os.path.join(image_output_dir, os.path.basename(imdb.image_path_at(i)))
            for j in range(len(rois)):
                box = rois[j]
                if probs[j][1] > 0.5:   #foreground region
                    cv2.rectangle(result_im, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))
                else:
                    cv2.rectangle(result_im, (box[0], box[1]), (box[2], box[3]), (255, 255, 255))
            cv2.imwrite(savename, result_im)
        _t['accuracy_detect'].toc()

        accuracys[i] = accuracy
        label_nums[i] = len(labels)
        right_num += accuracy * len(labels)
        total += len(labels)
        print 'accuracy_detect: {:d}/{:d} {:.3f}s, label_num: {:d}, accuracy: {:.3f}' \
              .format(i + 1, num_images, _t['accuracy_detect'].average_time, label_nums[i], float(accuracy))

    print 'Total Accuracy is:', right_num/total
    print 'RPN Cascade Accuracy Testing Finished!'

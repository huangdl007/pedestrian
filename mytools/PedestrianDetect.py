#!/usr/bin/env python

import _init_paths
import caffe
import cv2
import argparse
import numpy as np
import time, os, sys
import pprint

from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Pedestrian Detector')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default='models/mymodels/ZF/test.prototxt', type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default='output/zf_rfcn_end2end/voc_2007_trainval/zf_rfcn_iter_50000.caffemodel', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default='experiments/cfgs/rfcn_zf.yml', type=str)
    parser.add_argument('--thresh', dest='thresh',
                        help='Threshold for IOU',
                        default=0.5, type=float)
    parser.add_argument('--video', dest='videofile',
                        help='test video file',
                        default='data/testvideo/00.mp4', type=str)

    args = parser.parse_args()
    return args

class PedestrianDetector:
    def __init__(self, prototxt, caffemodel):
        caffe.set_mode_gpu()
        caffe.set_device(cfg.GPU_ID)
        self._net = caffe.Net(prototxt, caffemodel, caffe.TEST)
        # timers
        self._t = {'im_detect' : Timer(), 'misc' : Timer()}

    def detect(self, im=None, thresh=0.5):
        if im is None:
           return
        
        self._t['im_detect'].tic()

        #scores, boxes, landmarks = im_detect(self._net, im)
        scores, boxes = im_detect(self._net, im)
        self._t['im_detect'].toc()
        inds = np.where(scores[:, 1] > thresh)[0]
        cls_scores = scores[inds, 1]
        cls_boxes = boxes[inds, 4:8]
        #cls_landmarks = landmarks[inds, :]
        # Apply NMS
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
        keep = nms(cls_dets, cfg.TEST.NMS)
        det_boxes = cls_dets[keep, :]
        #det_landmarks = cls_landmarks[keep, :]

        for k in range(det_boxes.shape[0]):
            bbox = det_boxes[k][:4]
            score = det_boxes[k][-1]
            if score > thresh:
                cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0))
                cv2.putText(im, '{}: {:.3f}'.format('pedestrian', score), (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255))
                # paint the landmarks
                # lms = det_landmarks[k]
                # assert bbox[0]+lms[0] < bbox[2], 'landmarks must indside the box: '+ str(bbox) + str(lms)
                # cv2.circle(im, (bbox[0]+lms[0], bbox[1]+lms[1]), 3, (255, 0, 0), -1)
                # cv2.circle(im, (bbox[0]+lms[2], bbox[1]+lms[3]), 3, (0, 255, 0), -1)
                # cv2.circle(im, (bbox[0]+lms[4], bbox[1]+lms[5]), 3, (0, 0, 255), -1)

        
        print 'im_detect time: {:.3f}s'.format(self._t['im_detect'].average_time)

        return im

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    detector = PedestrianDetector(args.prototxt, args.caffemodel)

    capture = cv2.VideoCapture(args.videofile)
    num = 0
    if capture.isOpened():
        while True:
            ret, frame = capture.read()
            print 'frame:', num
            num += 1
            if ret == True:
                frame = cv2.resize(frame, (640, 360))
                frame = detector.detect(frame)
                cv2.imshow('video', frame)

                if cv2.waitKey(1) == 27:
                    break
            else:
                break

        cv2.destroyAllWindows()
    else:
        print 'open video fail!'




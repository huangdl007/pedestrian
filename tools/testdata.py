#!/usr/bin/env python

import pickle as pkl

with open('../data/cache/voc_2007_trainval_gt_roidb.pkl', 'r') as f1:
	gt_roidb = pkl.load(f1)

#print gt_roidb[2260:2280]
for roidb in gt_roidb:
	print len(roidb['gt_classes'])
	if len(roidb['gt_classes']) == 0:
		print roidb


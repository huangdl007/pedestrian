#/usr/bin/python

import cPickle as pk
import numpy as np
with open('output/rfcn_rpn_cascade_50_alt/voc_2007_test/stage1_resnet50_rpn_iter_50000_300_proposals.pkl') as ff:
	rois = pk.load(ff)

print rois[0]
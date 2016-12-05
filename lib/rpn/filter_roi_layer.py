import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg

DEBUG = False

class FilterRoiLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):

        # sampled rpn_rois (0, x1, y1, x2, y2)
        top[0].reshape(1, 5, 1, 1)
        
    def forward(self, bottom, top):

        all_rois = bottom[0].data

        probs = bottom[1].data.copy()
        #print probs
        probs = probs[:, 1]

        keep = np.where(probs > 0.5)[0]
        #print keep
        if len(keep) == 0:  #make sure have at least one roi
            keep = np.array([probs.argmax()])
            #print keep

        rois = all_rois[keep, :]
        #print 'filtered rois shape:', rois.shape
        top[0].reshape(*rois.shape)
        top[0].data[...] = rois

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


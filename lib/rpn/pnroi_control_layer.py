import caffe
import yaml
import numpy as np
import numpy.random as npr
from fast_rcnn.config import cfg


class PNroiControlLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):

        # sampled rpn_rois (0, x1, y1, x2, y2)
        top[0].reshape(1, 5, 1, 1)

        #labels
        top[1].reshape(1, 1, 1, 1)
        
    def forward(self, bottom, top):

        rois = bottom[0].data.copy()

        labels = bottom[1].data.copy()
        pos = np.where(labels == 1)[0]
        negs = np.where(labels == 0)[0]
        
        if len(pos) != 0 and len(negs) > 1.5*len(pos):
            negs = npr.choice(negs, size=1.5*len(pos))

        keep_inds = np.append(pos, negs)
        rois = rois[keep_inds]
        labels = labels[keep_inds]
        #print labels.shape, len(pos)

        top[0].reshape(*rois.shape)
        top[0].data[...] = rois

        top[1].reshape(*labels.shape)
        top[1].data[...] = labels

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


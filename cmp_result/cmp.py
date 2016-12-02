import os
import sys
lib_path = os.path.join('..', 'lib')
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
import cPickle
import numpy as np
from datasets.voc_eval import voc_eval
import matplotlib  
matplotlib.use('Agg')  
import matplotlib.pyplot as plt


classes = ('__background__', 'pedestrian')
devkit_path = '../data/VOCdevkit2007'
year = '2007'
image_set = 'test'

def load_image_set_index():
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
    # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
    image_set_file = os.path.join('../data/VOCdevkit2007/VOC2007', 'ImageSets', 'Main',
                                  image_set + '.txt')
    assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
        image_index = [x.strip() for x in f.readlines()]
    return image_index

def get_voc_results_file_template(resultType='old'):
    filename = resultType + '_det_' + '_{:s}.txt'
    path = os.path.join('tmp', filename)
    return path

def write_voc_results_file(all_boxes, resultType='old'):
    for cls_ind, cls in enumerate(classes):
        if cls == '__background__':
            continue
        filename = get_voc_results_file_template(resultType).format(cls)
        image_index = load_image_set_index()
        with open(filename, 'w') as f:
            for im_ind, index in enumerate(image_index):
                dets = all_boxes[cls_ind][im_ind]
                if dets == []:
                    continue
                for k in xrange(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))

def do_python_eval(resultType='old', thres=0.5):
    annopath = os.path.join(devkit_path, 'VOC' + year, 'Annotations', '{:s}.xml')
    imagesetfile = os.path.join(devkit_path, 'VOC' + year, 'ImageSets', 'Main',
                  image_set + '.txt')
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    #use_07_metric = False
    use_07_metric = True if int(year) < 2010 else False

    cls = 'pedestrian'

    filename = get_voc_results_file_template(resultType).format(cls)
    rec, prec, ap = voc_eval(
        filename, annopath, imagesetfile, cls, cachedir, ovthresh=thres,
        use_07_metric=use_07_metric)

    #print('Recall for {} = {:.4f}'.format(cls, rec[-1]))
    #print('Precision for {} = {:.4f}'.format(cls, prec[-1]))
    #print('AP for {} = {:.4f}'.format(cls, ap))

    return rec[-1], ap

def get_recall(thresholds, resultType='old'):
    detection_file = resultType + '_detections.pkl'
    with open(detection_file, 'rb') as f:
       all_boxes = cPickle.load(f)
    print len(all_boxes[0])
    write_voc_results_file(all_boxes, resultType)
    
    recalls = []
    aps = []
    for th in thresholds:
        rec, ap = do_python_eval(resultType, th)
        recalls.append(rec)
        aps.append(ap)
        print th, rec, ap

    return recalls, aps

if __name__ == '__main__':
    thresholds = np.arange(0.5, 1.0, 0.05)
    old_recalls, old_aps = get_recall(thresholds, 'old')
    new_recalls, new_aps = get_recall(thresholds, 'new')

    plt.figure()
    plt.plot(thresholds, old_recalls, label="old result",color="red",linewidth=2)
    plt.plot(thresholds, new_recalls, label="new result",color="blue",linewidth=2)
    plt.xlim(0.5,1.0)
    plt.savefig('recall curve.jpg')
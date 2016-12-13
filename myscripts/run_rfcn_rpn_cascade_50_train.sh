./tools/train_net.py --gpu 0 \
  --solver models/mymodels/rfcn_rpn_cascade_50/solver.prototxt \
  --weights output/rpn/voc_2007_trainval/resnet50_rpn_iter_50000.caffemodel \
  output/rpn_cascade/voc_2007_trainval/resnet50_rpn_cascade_iter_15000.caffemodel \
  --imdb voc_2007_trainval \
  --iters 50000 \
  --cfg experiments/cfgs/rfcn_rpn_cascade_50.yml

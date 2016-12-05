./tools/train_net.py --gpu 0 \
  --solver models/pascal_voc/ResNet-50/rfcn_end2end/solver_rpn_cascade.prototxt \
  --weights output/rpn_cascade/voc_2007_trainval/resnet50_rpn_cascade_iter_15000.caffemodel \
  --imdb voc_2007_trainval \
  --iters 50000 \
  --cfg experiments/cfgs/rfcn_end2end.yml

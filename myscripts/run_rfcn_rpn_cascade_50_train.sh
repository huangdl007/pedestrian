./tools/train_net.py --gpu 0 \
  --solver models/mymodels/rfcn_rpn_cascade_50/solver.prototxt \
  --weights output/rpn/voc_2007_trainval/resnet50_rpn_iter_50000.caffemodel \
  data/imagenet_models/ResNet-50-model.caffemodel \
  --imdb voc_2007_trainval \
  --iters 50000 \
  --cfg experiments/cfgs/rfcn_rpn_cascade_50.yml

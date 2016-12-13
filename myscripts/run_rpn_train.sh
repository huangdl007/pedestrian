./tools/train_net.py --gpu 0 \
  --solver models/mymodels/rpn/solver.prototxt \
  --weights data/imagenet_models/ResNet-50-model.caffemodel \
  --imdb voc_2007_trainval \
  --iters 50000 \
  --cfg experiments/cfgs/rpn.yml

./tools/train_net.py --gpu 0 \
  --solver models/mymodels/rfcn_rpn_cascade_50_alt/stage1_rpn_solver30k50k.prototxt \
  --weights data/imagenet_models/ResNet-50-model.caffemodel \
  --imdb voc_2007_trainval \
  --iters 50000 \
  --cfg experiments/cfgs/rpn.yml

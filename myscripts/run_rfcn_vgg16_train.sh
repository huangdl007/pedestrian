./tools/train_net.py --gpu 0 \
  --solver models/mymodels/VGG16/solver.prototxt \
  --weights data/imagenet_models/VGG16.caffemodel \
  --imdb voc_2007_trainval \
  --iters 50000 \
  --cfg experiments/cfgs/rfcn_vgg16.yml
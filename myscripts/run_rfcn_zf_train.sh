./tools/train_net.py --gpu 0 \
  --solver models/mymodels/ZF/solver.prototxt \
  --weights data/imagenet_models/ZF.caffemodel \
  --imdb voc_2007_trainval \
  --iters 50000 \
  --cfg experiments/cfgs/rfcn_zf.yml
./tools/train_net.py --gpu 0 \
  --solver models/mymodels/rfcn_rpn_cascade_50_alt/stage1_rfcn_solver.prototxt \
  --weights output/rfcn_rpn_cascade_50_alt/voc_2007_trainval/stage1_resnet50_rpn_cascade_iter_15000.caffemodel \
  --imdb voc_2007_trainval \
  --iters 50000 \
  --cfg experiments/cfgs/rfcn_rpn_cascade_50.yml

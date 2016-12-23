./tools/train_net.py --gpu 0 \
  --solver models/mymodels/rfcn_rpn_cascade_50_alt/stage2_rpn_solver30k50k.prototxt \
  --weights output/rfcn_rpn_cascade_50_alt/voc_2007_trainval/stage1_resnet50_rfcn_iter_30000.caffemodel \
  --imdb voc_2007_trainval \
  --iters 50000 \
  --cfg experiments/cfgs/rpn.yml

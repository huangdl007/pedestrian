./mytools/train_rpn_cascade.py --gpu 0 --solver models/mymodels/rfcn_rpn_cascade_50_alt/stage1_rpn_cascade_solver.prototxt --weights output/rfcn_rpn_cascade_50_alt/voc_2007_trainval/stage1_resnet50_rpn_iter_50000.caffemodel --imdb voc_2007_trainval --iters 15000 --cfg experiments/cfgs/rpn_cascade.yml

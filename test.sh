nohup python test_models.py Flow ./models/flow/virat_bninception_flow_flow_model_best.pth.tar \
	   --arch BNInception --flow_pref flow_ --mode val --gpus 1 --test_segments 25 --save_scores ./nohups/res/test_finetune_flow_bn_seg5_grad_all.csv \
> nohups/test_finetune_flow_bn_seg5_grad_all.out 2>&1 &

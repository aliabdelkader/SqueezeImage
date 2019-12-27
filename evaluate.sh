#!/bin/bash
export CUDA_VISIBLE_DEVICES="5"
export model_name="SqueezeImage"
sudo /home/mrafaat/anaconda3/envs/myenv/bin/python CityscapesEvaluator.py \
--dataset_root_path "/raid/ali/cityscapes" \
--imageset_path "cityscapes/imagesets" \
--dataset_config_path "cityscapes/cityscapes.yaml" \
--results_dir "/raid/ali/results/${model_name}" \
--device "cuda" \
--image_height "512" \
--image_width "1024" \
--model_name "${model_name}" \
--model_path "results/SqueezeImage/best_99.pth" \
--output_classes "20" \

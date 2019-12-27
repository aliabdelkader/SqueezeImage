#!/bin/bash
export CUDA_VISIBLE_DEVICES="5"
export model_name="SqueezeImage"
python CityscapesEvaluator.py \
--dataset_root_path "/mnt/sdb1/ali/cityscapes" \
--imageset_path "cityscapes/imagesets" \
--dataset_config_path "cityscapes/cityscapes.yaml" \
--results_dir "/mnt/sdb1/ali/results/${model_name}" \
--device "cuda" \
--image_height "512" \
--image_width "1024" \
--model_name "${model_name}" \
--model_path "results/SqueezeImage/best_99.pth" \
--output_classes "20" \

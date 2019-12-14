#!/bin/bash
# export CUDA_LAUNCH_BLOCKING="1"
export CUDA_VISIBLE_DEVICES="6,7"
export model_name="SqueezeImage"
python main.py \
--model_name "${model_name}" \
--dataset_root_path "/mnt/sdb1/ali/cityscapes" \
--dataset_config_path "cityscapes/cityscapes.yaml" \
--batch_size 64 \
--imageset_path "cityscapes/imagesets" \
--results_dir "results/${model_name}" \
--learning_rate "0.001" \
--number_epochs "100" \
--image_height "256" \
--image_width "512" \
--output_classes "20" \
--device "cuda" \
--num_gpus "2"

#!/bin/bash
# export CUDA_LAUNCH_BLOCKING="1"
#export CUDA_VISIBLE_DEVICES="0,1"
#export model_name="SqueezeImage"
#python main.py \
#--model_name "${model_name}" \
#--dataset_root_path "/raid/ali/cityscapes" \
#--dataset_config_path "cityscapes/cityscapes.yaml" \
#--batch_size 10 \
#--imageset_path "cityscapes/imagesets" \
#--results_dir "results/${model_name}" \
#--learning_rate "0.001" \
#--number_epochs "200" \
#--image_height "512" \
#--image_width "1024" \
#--output_classes "20" \
#--device "cuda" \
#--num_gpus "2"

export CUDA_VISIBLE_DEVICES="0,1"
export model_name="SqueezeImage"
python main.py \
--model_name "${model_name}" \
--dataset_root_path "/home/fusionresearch/Datasets/cityscapes" \
--dataset_config_path "cityscapes/cityscapes.yaml" \
--batch_size 10 \
--imageset_path "cityscapes/imagesets" \
--results_dir "results/${model_name}" \
--learning_rate "0.001" \
--number_epochs "200" \
--image_height "512" \
--image_width "1024" \
--output_classes "20" \
--device "cuda" \
--num_gpus "2"

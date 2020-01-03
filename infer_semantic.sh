#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python SemanticIKitti_infer.py \
--dataset_root_path "/home/fusionresearch/SemanticKitti/dataset" \
--imageset_path "semantickitti/imageset" \
--results_dir $(pwd)'/SemanticKitti' \
--device 'cuda' \
--model_name "SqueezeImage" \
--model_path "results/SqueezeImage/best_179.pth" \
--output_classes '20'

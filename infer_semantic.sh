#!/bin/bash

python SemanticKitti_infer \
--dataset_root_path "/home/fusionresearch/SemanticKitti/dataset" \
--imageset_path "semantickitti/imageset" \
--results_dir 'SemanticKitti' \
--device 'cuda' \
--model_name "SqueezeImage" \
--model_path "resutls/SqueezeImage/best_179.pth" \
--output_classes '20'
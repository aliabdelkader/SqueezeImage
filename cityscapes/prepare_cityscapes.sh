#python create_imagesets.py --dataset_root "/raid/ali/cityscapes" --output_dir "/home/mrafaat/ali/SqueezeImage/cityscapes/imagesets"
#python calculate_stats.py \
#--dataset_root_path "/raid/ali/cityscapes" \
#--imageset_path "imagesets" \
#--dataset_config_path "cityscapes.yaml" \
#--image_height "512" \
#--image_width "1024" \
#--output_classes "20"
export CITYSCAPES_DATASET="/home/fusionresearch/Datasets/cityscapes"
cd cityscapesScripts
python cityscapesscripts/preparation/createTrainIdLabelImgs.py
cd ..
python create_imagesets.py --dataset_root "/home/fusionresearch/Datasets/cityscapes" --output_dir "/home/fusionresearch/ali/SqueezeImage/cityscapes/imagesets"
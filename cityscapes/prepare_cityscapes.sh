#python create_imagesets.py --dataset_root "/mnt/sdb1/ali/cityscapes" --output_dir "/home/mrafaat/ali/SqueezeImage/cityscapes/imagesets"
python cityscapes/calculate_stats.py \
--dataset_root_path "/raid/ali/cityscapes" \
--imageset_path "cityscapes/imagesets" \
--dataset_config_path "cityscapes/cityscapes.yaml" \
--image_height "512" \
--image_width "1024" \
--output_classes "20"

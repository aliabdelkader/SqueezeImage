"""
script to create train, val, test images
expected dataset folder tree

+ cityscapes
    + gFine
        + train
        + test
        + val
            + city
                + *_labelTrainIds
"""

import argparse
from pathlib import Path
from tqdm import tqdm



def create_imageset(annotations_path: Path, imageset_name: str) -> list:
    """
    function creates list of images "imageset" from given path

    Args:
        annotations_path: path to annontation files i.e cityscapes/gFine
        imageset_name: imageset name i.e: train

    Returns:
        list of filenames
    """
    imageset = []
    annotation_images_path = (annotations_path/imageset_name).rglob("*_gtFine_labelTrainIds.png")
    for annotation_image_path in tqdm(annotation_images_path, "processing imageset {}".format(imageset_name)):
        annotation_image_name = annotation_image_path.stem.replace("_gtFine_labelTrainIds", "")
        annotation_image_name_splitted = annotation_image_name.split("_")
        city = annotation_image_name_splitted[0]
        filename_in_imageset_name = "{}/{}".format(city, annotation_image_name)
        imageset.append(filename_in_imageset_name)
    return imageset

def write_imageset_disk(output: Path, imageset_name: str, imageset: list):
    """
    function writes imageset to disk

    Args:
        imageset: list of file names
        output: path to write imagesets
        imageset_name: name of imageset i.e train
    """
    imageset_path = output / (imageset_name + "*.txt")
    with open(str(imageset_path), 'w') as out:
        for i in imageset:
            out.write(i + '\n')

def main():
    parser = argparse.ArgumentParser(description='prepare imageset')
    parser.add_argument('--dataset_root', default="/home/fusionresearch/Datasets/cityscapes")
    parser.add_argument('--output_dir', default="../imageset")

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir  = Path(args.output_dir)

    gFine_path = dataset_root / "gFine"
    output_dir.mkdir(parents=True, exist_ok=True)

    # list of train, val, test file sets
    trainset = create_imageset(gFine_path, "train")
    write_imageset_disk(output=output_dir, imageset_name="train", imageset=trainset)

    valset = create_imageset(gFine_path, "val")
    write_imageset_disk(output=output_dir, imageset_name="val", imageset=valset)

    testset = create_imageset(gFine_path, "test")
    write_imageset_disk(output=output_dir, imageset_name="test", imageset=testset)

    print("Done creating imagesets")

if __name__ == "__main__":
    main()

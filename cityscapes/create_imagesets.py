"""
script to create train, val, test images
expected dataset folder tree

+ cityscapes
    + gtCoarse
        + train
        + train_extra
        + val
    + gFine
        + train
        + test
        + val
            + city
                + *_labelTrainIds
    + LeftImage
        + test
        + train
        + train_extra
        + val
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

        {root}/{type}{video}/{split}/{city}/{city}_{seq:0>6}_{frame:0>6}_{type}{ext}

    Returns:
        list of filenames
    """
    imageset = []
    annotation_images_path = (annotations_path / imageset_name).rglob("*_labelTrainIds.png")
    for annotation_image_path in tqdm(annotation_images_path, "processing imageset {}".format(imageset_name)):
        # annotation_image_name = annotation_image_path.stem.replace("_gtFine_labelTrainIds", "")
        city = annotation_image_path.parent.stem
        split = annotation_image_path.parent.parent.stem
        type = annotation_image_path.parent.parent.parent.stem
        annotation_image_name_splitted = annotation_image_path.stem.split("_")
        filename_in_imageset_name = "{s}-{c}-{fa}-{fb}-{t}".format(s=split, c=city, fa=annotation_image_name_splitted[1], fb=annotation_image_name_splitted[2], t=type)
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
    imageset_path = output / (imageset_name + ".txt")
    with open(str(imageset_path), 'w') as out:
        for i in imageset:
            out.write(i + '\n')


def main():
    parser = argparse.ArgumentParser(description='prepare imageset')
    parser.add_argument('--dataset_root', default="/home/fusionresearch/Datasets/cityscapes")
    parser.add_argument('--output_dir', default="../imageset")

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)

    gFine_path = dataset_root / "gtFine"
    gtCorse_path = dataset_root / "gtCoarse"
    output_dir.mkdir(parents=True, exist_ok=True)

    # list of train, val, test file sets
    trainset = create_imageset(gFine_path, "train")
    trainset.extend(create_imageset(gtCorse_path, "train"))
    trainset.extend(create_imageset(gtCorse_path, "train_extra"))
    write_imageset_disk(output=output_dir, imageset_name="train", imageset=trainset)

    valset = create_imageset(gFine_path, "val")
    valset.extend(create_imageset(gtCorse_path, "val"))
    write_imageset_disk(output=output_dir, imageset_name="val", imageset=valset)

    testset = create_imageset(gFine_path, "test")
    write_imageset_disk(output=output_dir, imageset_name="test", imageset=testset)

    print("Done creating imagesets")


if __name__ == "__main__":
    main()

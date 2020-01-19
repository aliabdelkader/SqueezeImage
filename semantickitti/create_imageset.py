"""
script to create train, val, test images
expected dataset folder tree


"""

import argparse
from pathlib import Path
from tqdm import tqdm



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
    parser.add_argument('--dataset_root', default="~/SemanticKitti/dataset/")

    parser.add_argument('--output_dir', default="imageset")

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)

    sequences_path = dataset_root / "sequences"
    output_dir.mkdir(parents=True, exist_ok=True)

    # sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
    #              "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
    sequences = ["08"]

    imageset = []
    for seq in sequences:
        images_in_seq = (sequences_path / seq / "image_2").rglob("*.png")
        images_in_seq_str = [str(i) for i in images_in_seq]
        imageset.extend(images_in_seq_str)
    # list of train, val, test file sets

    write_imageset_disk(output=output_dir, imageset_name="all", imageset=imageset)



    print("Done creating imagesets")


if __name__ == "__main__":
    main()

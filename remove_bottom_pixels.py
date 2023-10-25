import argparse
import os
from multiprocessing.pool import Pool

import cv2
from tqdm import tqdm

from page_xml.xmlPAGE import PageData
from utils.input_utils import get_file_paths
from utils.path_utils import image_path_to_xml_path


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main file for Layout Analysis")

    detectron2_args = parser.add_argument_group("detectron2")

    detectron2_args.add_argument("-c", "--config", help="config file", required=True)
    detectron2_args.add_argument("--opts", nargs="+", action="extend", help="optional args to change", default=[])

    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input folder/file", nargs="+", action="extend", type=str, required=True)
    args = parser.parse_args()

    return args


def remove_pixels(image_path):
    image_output_path = ""  # TODO
    image = cv2.imread(image_path)
    output_image = image[-15:]
    cv2.imwrite(image_output_path, output_image)


def main(args: argparse.Namespace):
    image_formats = [
        ".bmp",
        ".dib",
        ".jpeg",
        ".jpg",
        ".jpe",
        ".jp2",
        ".png",
        ".webp",
        ".pbm",
        ".pgm",
        ".ppm",
        ".pxm",
        ".pnm",
        ".pfm",
        ".sr",
        ".ras",
        ".tiff",
        ".tif",
        ".exr",
        ".hdr",
        ".pic",
    ]
    image_paths = get_file_paths(args.input, image_formats)

    # Single threaded
    # for image_path in tqdm(image_paths):
    #     generate_empty_page_xml(image_path)

    # Multi threading
    with Pool(os.cpu_count()) as pool:
        _ = list(tqdm(pool.imap_unordered(remove_pixels, image_paths), total=len(image_paths)))


if __name__ == "__main__":
    args = get_arguments()
    main(args)

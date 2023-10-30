import argparse
import logging
import os
import random
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
from tqdm import tqdm

from utils.input_utils import clean_input_paths, get_file_paths
from utils.logging_utils import get_logger_name


class Creator:
    def __init__(self, input_paths=None, output_dir=None, copy=False) -> None:
        self.logger = logging.getLogger(get_logger_name())

        self.input_paths: Optional[Sequence[Path]] = None
        if input_paths is not None:
            self.set_input_paths(input_paths)

        self.output_dir: Optional[Path] = None
        if output_dir is not None:
            self.set_output_dir(output_dir)

        self.copy = copy

        self.image_formats = [
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

    def set_input_paths(
        self,
        input_paths: str | Path | Sequence[str | Path],
    ) -> None:
        """
        Setter for image paths, also cleans them to be a list of Paths

        Args:
            input_paths (str | Path | Sequence[str | Path]): path(s) from which to extract the images

        Raises:
            FileNotFoundError: input path not found on the filesystem
            PermissionError: input path not accessible
        """
        input_paths = clean_input_paths(input_paths)

        all_input_paths = []

        for input_path in input_paths:
            if not input_path.exists():
                raise FileNotFoundError(f"Input ({input_path}) is not found")

            if not os.access(path=input_path, mode=os.R_OK):
                raise PermissionError(f"No access to {input_path} for read operations")

            input_path = input_path.resolve()
            all_input_paths.append(input_path)

        self.input_paths = all_input_paths

    def set_output_dir(self, output_dir: str | Path) -> None:
        """
        Setter for the output dir

        Args:
            output_dir (str | Path): path to output dir
        """
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)

        if not output_dir.is_dir():
            self.logger.info(f"Could not find output dir ({output_dir}), creating one at specified location")
            output_dir.mkdir(parents=True)

        self.output_dir = output_dir.resolve()

    def overlay_images_random_transform(self, foreground_path, background_path):
        foreground_image = cv2.imread(foreground_path)
        background_image = cv2.imread(background_path)

        foreground_height, foreground_width = foreground_image.shape[:2]
        background_height, background_width = background_image.shape[:2]

        max_scale_factor = min([background_height / foreground_height, background_width / foreground_width])
        scale_factor = np.random.uniform(max_scale_factor * 0.25, max_scale_factor)
        scaled_foreground_image = cv2.resize(foreground_image, None, fx=scale_factor, fy=scale_factor)
        scaled_foreground_height, scaled_foreground_width = scaled_foreground_image.shape[:2]

        max_trans_x = background_width - scaled_foreground_width
        max_trans_y = background_height - scaled_foreground_height
        trans_x = random.randint(0, max_trans_x)
        trans_y = random.randint(0, max_trans_y)

        angle = random.uniform(-30, 30)
        M = cv2.getRotationMatrix2D((scaled_foreground_width / 2, scaled_foreground_height / 2), angle, 1)

        M[:, 2] += [trans_x, trans_y]
        image_affine = cv2.warpAffine(scaled_foreground_image, M, (background_width, background_height))
        mask = cv2.warpAffine(np.ones_like(scaled_foreground_image) * 255, M, (background_width, background_height))

        image_result = background_image.copy()
        image_result[np.nonzero(mask)] = image_affine[np.nonzero(mask)]

        corners = cv2.transform(
            np.array(
                [
                    [
                        [0, 0],
                        [0, scaled_foreground_height],
                        [scaled_foreground_width, 0],
                        [scaled_foreground_width, scaled_foreground_height],
                    ]
                ],
                dtype=np.float32,
            ),
            M,
        )
        corners = corners.squeeze().astype(int)

        # for corner in corners:
        #     image = cv2.circle(image_result, corner, 5, (255, 0, 0), -1)

        if np.any(np.logical_or(0 > corners[:, 0], corners[:, 0] > background_width)) or np.any(
            np.logical_or(0 > corners[:, 1], corners[:, 1] > background_height)
        ):
            self.logger.warning(f"Corner out of bounds. Corners: {corners} Bounds: {background_width, background_height}")

        output_path = "test.jpg"

        cv2.imwrite(output_path, image_result)

        return corners

    def create_page(self, image):
        pass

    def process(self):
        if self.input_paths is None:
            raise TypeError("Cannot run when the input_paths is None")

        image_paths = get_file_paths(self.input_paths, self.image_formats)

        # Single threaded
        # for image_path in tqdm(image_paths):
        #     generate_empty_page_xml(image_path)

        # Multi threading
        with Pool(os.cpu_count()) as pool:
            _ = list(tqdm(pool.imap_unordered(self.create_page, image_paths), total=len(image_paths)))


def main(args: argparse.Namespace):
    creator = Creator(
        input_paths=args.input,
        output_dir=args.output,
        copy=args.copy,
    )
    creator.process()


if __name__ == "__main__":
    creator = Creator()
    creator.overlay_images_random_transform(
        "/home/stefan/Documents/SURFdrive/Shared/ovdr_images/photo/0b74d52a-5601-6794-9b98-0101b4de8777.jpg",
        "/home/stefan/Documents/SURFdrive/Shared/ovdr_images/document/0a04f473-3586-faa3-ca9a-874e2d6231ff.jpg",
    )

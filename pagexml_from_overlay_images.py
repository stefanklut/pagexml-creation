import argparse
import logging
import os
import random
import uuid
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
from tqdm import tqdm

from page_xml.xmlPAGE import PageData
from utils.input_utils import get_file_paths, supported_image_formats
from utils.logging_utils import get_logger_name
from utils.path_utils import image_path_to_xml_path


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main file for Layout Analysis")

    io_args = parser.add_argument_group("IO")
    io_args.add_argument(
        "-f", "--foreground", help="Foreground folder/file", nargs="+", action="extend", type=str, required=True
    )
    io_args.add_argument(
        "-b", "--background", help="Foreground folder/file", nargs="+", action="extend", type=str, required=True
    )
    io_args.add_argument("-o", "--output", help="Output folder", type=str)

    parser.add_argument("-n", "--number", type=int, help="Number of images generated")

    args = parser.parse_args()

    return args


class Creator:
    def __init__(
        self,
        foreground_paths=None,
        background_paths=None,
        output_dir=None,
        number=None,
        copy=False,
    ) -> None:
        self.logger = logging.getLogger(get_logger_name())

        self.foreground_paths: Optional[Sequence[Path]] = None
        if foreground_paths is not None:
            self.set_foreground_paths(foreground_paths)

        self.background_paths: Optional[Sequence[Path]] = None
        if background_paths is not None:
            self.set_background_paths(background_paths)

        self.output_dir: Optional[Path] = None
        if output_dir is not None:
            self.set_output_dir(output_dir)

        self._random_foreground_paths = []
        self._random_background_paths = []

        self.number = number

        self.copy = copy

    def set_foreground_paths(
        self,
        foreground_paths: str | Path | Sequence[str | Path],
    ) -> None:
        """
        Setter for image paths, also cleans them to be a list of Paths

        Args:
            input_paths (str | Path | Sequence[str | Path]): path(s) from which to extract the images

        Raises:
            FileNotFoundError: input path not found on the filesystem
            PermissionError: input path not accessible
        """
        self.foreground_paths = get_file_paths(foreground_paths, supported_image_formats)

    def set_background_paths(
        self,
        background_paths: str | Path | Sequence[str | Path],
    ) -> None:
        """
        Setter for image paths, also cleans them to be a list of Paths

        Args:
            input_paths (str | Path | Sequence[str | Path]): path(s) from which to extract the images

        Raises:
            FileNotFoundError: input path not found on the filesystem
            PermissionError: input path not accessible
        """
        self.background_paths = get_file_paths(background_paths, supported_image_formats)

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

    @staticmethod
    def within_rectangle(points, rectangle):
        if np.any(rectangle[0, 0] > points[:, 0]):
            return False
        if np.any(points[:, 0] > rectangle[1, 0]):
            return False
        if np.any(rectangle[0, 1] > points[:, 1]):
            return False
        if np.any(points[:, 1] > rectangle[1, 1]):
            return False

        return True

    def overlay_images_random_transform(self, foreground_image: np.ndarray, background_image: np.ndarray):
        foreground_height, foreground_width = foreground_image.shape[:2]
        background_height, background_width = background_image.shape[:2]

        max_scale_factor = min([background_height / foreground_height, background_width / foreground_width])
        scale_factor = np.random.uniform(max_scale_factor * 0.25, max_scale_factor)
        scaled_foreground_image = cv2.resize(foreground_image, None, fx=scale_factor, fy=scale_factor)
        scaled_foreground_height, scaled_foreground_width = scaled_foreground_image.shape[:2]

        max_trans_x = background_width - scaled_foreground_width
        max_trans_y = background_height - scaled_foreground_height

        max_iters = 1000

        for i in range(max_iters):
            trans_x = random.randint(0, max_trans_x)
            trans_y = random.randint(0, max_trans_y)
            angle = random.uniform(-30, 30)
            M = cv2.getRotationMatrix2D((scaled_foreground_width / 2, scaled_foreground_height / 2), angle, 1)

            M[:, 2] += [trans_x, trans_y]

            corners = cv2.transform(
                np.array(
                    [
                        [
                            [0, 0],
                            [scaled_foreground_width, 0],
                            [scaled_foreground_width, scaled_foreground_height],
                            [0, scaled_foreground_height],
                        ]
                    ],
                    dtype=np.float32,
                ),
                M,
            )
            corners = corners.squeeze().astype(int)

            if self.within_rectangle(corners, np.array([(0, 0), (background_width, background_height)])):
                break
        else:
            raise ValueError(f"No valid image location found within {max_iters} tries")

        image_affine = cv2.warpAffine(scaled_foreground_image, M, (background_width, background_height))
        mask = cv2.warpAffine(np.ones_like(scaled_foreground_image) * 255, M, (background_width, background_height))

        image_result = background_image.copy()
        image_result[np.nonzero(mask)] = image_affine[np.nonzero(mask)]

        output_path = "test.jpg"

        return image_result, corners

    def random_foreground_path(self):
        if self.foreground_paths is None:
            raise TypeError("Cannot run when the foreground_paths is None")

        if len(self._random_foreground_paths) == 0:
            self._random_foreground_paths.extend(self.foreground_paths)
            random.shuffle(self._random_foreground_paths)
        return self._random_foreground_paths.pop()

    def random_background_path(self):
        if self.background_paths is None:
            raise TypeError("Cannot run when the foreground_paths is None")

        if len(self._random_background_paths) == 0:
            self._random_background_paths.extend(self.background_paths)
            random.shuffle(self._random_background_paths)
        return self._random_background_paths.pop()

    def create_page(self, items):
        if self.output_dir is None:
            raise TypeError("Cannot run when the output_dir is None")
        foreground_path, background_path, i = items
        foreground_image = cv2.imread(str(foreground_path))

        background_image = cv2.imread(str(background_path))

        image, corners = self.overlay_images_random_transform(foreground_image, background_image)

        image_path = self.output_dir.joinpath(f"{i}.jpg")

        cv2.imwrite(str(image_path), image)

        page_dir = self.output_dir.joinpath("page")

        if not page_dir.is_dir():
            self.logger.info(f"Could not find page dir ({page_dir}), creating one at specified location")
            page_dir.mkdir(parents=True)

        xml_path = image_path_to_xml_path(image_path, check=False)

        page = PageData(xml_path)
        page.new_page(image_path.name, str(image.shape[0]), str(image.shape[1]))

        region_coords = ""
        for coords in corners.reshape(-1, 2):
            region_coords = region_coords + f" {coords[0]},{coords[1]}"
        region_coords = region_coords.strip()

        region_type = "ImageRegion"
        region = "Photo"
        region_id = 1

        _uuid = uuid.uuid4()
        image_reg = page.add_element("ImageRegion", f"region_{_uuid}_{region_id}", region, region_coords)

        page.save_xml()

    def process(self):
        if self.foreground_paths is None:
            raise TypeError("Cannot run when the foreground_paths is None")

        if self.background_paths is None:
            raise TypeError("Cannot run when the background_paths is None")

        if self.output_dir is None:
            raise TypeError("Cannot run when the output_dir is None")

        if self.number is None:
            raise TypeError("Cannot run when the number is None")

        items = [(self.random_foreground_path(), self.random_background_path(), i) for i in range(self.number)]

        # Single threaded
        # for image_path in tqdm(image_paths):
        #     generate_empty_page_xml(image_path)

        # Multi threading
        with Pool(os.cpu_count()) as pool:
            _ = list(tqdm(pool.imap_unordered(self.create_page, items), total=self.number))


def main(args: argparse.Namespace):
    creator = Creator(
        foreground_paths=args.foreground,
        background_paths=args.background,
        output_dir=args.output,
        number=args.number,
    )
    creator.process()


if __name__ == "__main__":
    args = get_arguments()
    main(args)

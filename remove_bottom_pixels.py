import argparse
import logging
import os
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Optional, Sequence

import cv2
from tqdm import tqdm

from utils.input_utils import get_file_paths, supported_image_formats
from utils.logging_utils import get_logger_name


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main file for Layout Analysis")

    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input folder/file", nargs="+", action="extend", type=str, required=True)
    io_args.add_argument("-o", "--output", help="Output folder", type=str, required=True)

    args = parser.parse_args()

    return args


class Creator:
    def __init__(self, input_paths=None, output_dir=None) -> None:
        self.logger = logging.getLogger(get_logger_name())

        self.input_paths: Optional[Sequence[Path]] = None
        if input_paths is not None:
            self.set_input_paths(input_paths)

        self.output_dir: Optional[Path] = None
        if output_dir is not None:
            self.set_output_dir(output_dir)

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
        self.input_paths = get_file_paths(input_paths, supported_image_formats)

    # TODO Set multiple output paths
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

    def remove_bottom_pixels(self, image_path: Path):
        if self.output_dir is None:
            raise TypeError("Output dir is None")
        output_path = self.output_dir.joinpath(image_path.name)
        image = cv2.imread(str(image_path))
        output_image = image[:-15]
        cv2.imwrite(str(output_path), output_image)

    def process(self):
        if self.input_paths is None:
            raise TypeError("Cannot run when the input_paths is None")

        # Single threaded
        # for image_path in tqdm(image_paths):
        #     remove_bottom_pixels(image_path)

        # Multi threading
        with Pool(os.cpu_count()) as pool:
            _ = list(tqdm(pool.imap_unordered(self.remove_bottom_pixels, self.input_paths), total=len(self.input_paths)))


def main(args: argparse.Namespace):
    creator = Creator(
        input_paths=args.input,
        output_dir=args.output,
    )
    creator.process()


if __name__ == "__main__":
    args = get_arguments()
    main(args)

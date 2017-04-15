
from lib.config import cfg

from random import shuffle
import cv2

import os


class ImageLoader(object):
    def __init__(self, folder_path):
        self._folder_path = folder_path
        self._files = self._read_files(folder_path)
        self._images_num = len(self._files)
        self._curr_index = 0

    def next_image(self):
        """
        Load the next image in a batch.
        """
        index = self._curr_index
        self._curr_index += 1
        if self._curr_index >= len(self._files):
            shuffle(self._files)
            self._curr_index = 0

        im_name = self._files[index]
        im_path = os.path.join(self._folder_path, self._files[index])
        im = cv2.imread(im_path)
        im = cv2.resize(im, (cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH), interpolation=cv2.INTER_CUBIC)

        return (im, im_name, im_path)

    @property
    def size(self):
        return self._images_num

    def _read_files(self, folder_path):
        """
        Read image hashes.
        :param folder_path: Path to image folder.
        :return: List of file paths.
        """
        try:
            file_paths = [file for file in os.listdir(folder_path)]

            if len(file_paths) == 0:
                print("Error while loading data. Add some images to folder 'dataset/data' !")
                exit()
            return file_paths
        except:
            print("Error while loading data.")
            exit()
from PIL import Image
import numpy as np
from os.path import join


class PreProcessor:
    """
    The class built aims to process any size of
    pictures and prevent it from occupy cuda
    memory too much to get broken.
    """
    def __init__(self, patch_width: int = 600, patch_height: int = 400):
        self.to_work = True
        self.container = []
        self.patch_width = patch_width
        self.patch_height = patch_height

    def __call__(self, img) -> list:
        if type(img) == str:
            real_img = Image.open(img)
        else:
            real_img = img

        self.img = np.array(real_img)
        width, height = real_img.size
        self.max_width = width
        self.max_height = height

        if self.max_width <= 600 and self.max_height <= 400:
            return [self.img]
        else:
            return self.extract_patches()

    def name(self):
        return 'PreProcessor'

    def get_patch_num(self, img_size: tuple) -> tuple:
        img_height = img_size[0]
        img_width = img_size[1]
        steps_along_width = int(self.max_width / self.patch_width + 1)
        steps_along_height = int(self.max_height / self.patch_height + 1)
        return (steps_along_width, ) + (steps_along_height, )

    def clear_container(self):
        self.container = []

    def add_to_container(self, img_patch: np.ndarray):
        self.container.append(img_patch)

    def resize_for_deploy(self, img):
        resized_img = img.reshape(750, 400)
        return resized_img

    def save_the_whole_image(self, img, defaul_path='./', default_name='pic.jpg'):
        img.save(join(defaul_path, default_name))
        print(f"The image {join(defaul_path, default_name)} has been saved!")

    def extract_patches(self):
        patch_line_num, patch_column_num = self.get_patch_num(self.img.shape)
        for line in range(0, patch_line_num):
            for column in range(0, patch_column_num):
                self.add_to_container(
                    self.img[
                        column * self.patch_height: min(self.max_height, (column + 1) * self.patch_height),
                        line * self.patch_width: min(self.max_width, (line + 1) * self.patch_width),
                        :
                    ])
        return self.container

    def restore_picture(self):
        tmp_img_container = np.zeros((self.max_height, self.max_width, 3))
        patch_line_num, patch_column_num = self.get_patch_num(self.img.shape)
        index = 0
        for line in range(0, patch_line_num):
            for column in range(0, patch_column_num):
                tmp_img_container[
                    column * self.patch_height: min(self.max_height, (column + 1) * self.patch_height),
                    line * self.patch_width: min(self.max_width, (line + 1) * self.patch_width),
                    :
                ] = self.container[index]
                index += 1
        # restored_img = Image.fromarray(np.unit8(tmp_img_container).convert('RGB'))
        restored_img = Image.fromarray(tmp_img_container.astype('uint8'), 'RGB')
        return restored_img

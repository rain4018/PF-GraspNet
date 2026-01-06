import glob
import os
import cv2
import numpy as np

from utils.dataset_processing import grasp
from .grasp_data import GraspDatasetBase
from skimage.transform import rotate


def normalise_depth_img(image):
    valid_mask = (image > 0)

    d_min = image[valid_mask].min()
    d_max = image[valid_mask].max()

    depth_img_norm = np.zeros_like(image, dtype=np.float32)
    depth_img_norm[valid_mask] = (image[valid_mask] - d_min) / (d_max - d_min)

    return depth_img_norm


def padding(image, target_size=(640, 640), is_depth=False, pad_value=None):
    original_h, original_w = image.shape[:2]
    target_h, target_w = target_size

    pad_w = target_w - original_w
    pad_h = target_h - original_h

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if pad_value is None:
        pad_value = 0 if is_depth else (114, 114, 114)

    if len(image.shape) == 2:
        padded = cv2.copyMakeBorder(image, top, bottom, left, right,
                                    borderType=cv2.BORDER_CONSTANT, value=pad_value)
    else:
        padded = cv2.copyMakeBorder(image, top, bottom, left, right,
                                    borderType=cv2.BORDER_CONSTANT, value=pad_value)
    return padded


def resize_image(image, target_size, is_depth=False, factor=1.0):
    original_h, original_w = image.shape[:2]
    target_h, target_w = target_size
    target_h = target_h * factor
    target_w = target_w * factor

    scale = min(target_w / original_w, target_h / original_h)
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)

    interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    return resized


def rotate_image(image, angle, center=None):
    if center is not None:
        center = (center[1], center[0])
    image = rotate(image, angle / np.pi * 180, center=center, mode='symmetric', preserve_range=True).astype(
        image.dtype)

    return image


class CBRGDDataset(GraspDatasetBase):
    def __init__(self, file_path, scene_id, ds_rotate=0, split='train', **kwargs):
        super(CBRGDDataset, self).__init__(**kwargs)
        if split == 'train':
            self.grasp_files = glob.glob(os.path.join(file_path, str(scene_id), 'source', '*', '*cpos.txt'))
        else:
            self.grasp_files = glob.glob(os.path.join(file_path, str(scene_id), 'target', '*', '*cpos.txt'))

        self.grasp_files.sort()
        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]

        self.depth_files = [f.replace('cpos.txt', 'd.png') for f in self.grasp_files]
        self.rgb_files = [f.replace('d.png', '.png') for f in self.depth_files]

    def get_gtbb(self, idx, rot=0):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        gtbbs.offset((80, 0))

        gtbbs.rotate(rot, (320, 320))

        scale_factor = self.output_size / 640
        gtbbs.zoom(1 / scale_factor, (320, 320))

        return gtbbs

    def get_depth(self, idx, rot=0):
        depth_img = cv2.imread(self.depth_files[idx], cv2.IMREAD_UNCHANGED).astype(np.float32)

        height, width = depth_img.shape[:2]
        size = height if height >= width else width

        depth_img = padding(depth_img, (size, size), is_depth=True, pad_value=0)
        depth_img = rotate_image(depth_img, rot)
        depth_img = normalise_depth_img(depth_img)
        depth_img = resize_image(depth_img, (self.output_size, self.output_size), is_depth=True)

        return depth_img

    def get_rgb(self, idx, rot=0, normalise=True):
        rgb_img = cv2.imread(self.rgb_files[idx])

        height, width = rgb_img.shape[:2]
        size = height if height >= width else width

        rgb_img = padding(rgb_img, (size, size))

        rgb_img = rotate_image(rgb_img, rot)
        rgb_img = resize_image(rgb_img, (self.output_size, self.output_size))

        if normalise:
            rgb_img = rgb_img.astype(np.float32) / 255.0
            rgb_img = np.transpose(rgb_img, (2, 0, 1))

        return rgb_img

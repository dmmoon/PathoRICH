from __future__ import print_function, division
from torch.utils.data import IterableDataset
import cv2

import numpy as np

from typing import Any

def get_tissue_mask(image):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    mask = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return mask



class PatchGenerator(IterableDataset):
    def __init__(self, opt: object, wsi: object, transform: Any):
        super(PatchGenerator).__init__()
        self.opt = opt
        self.wsi = wsi
        # self.slide_io = slide_io
        self.transform_fn = transform
        self.tissue_mask = self.get_tissue_mask()
        self.data = self.get_patch_area_list()

    def get_tissue_mask(self):
        # wsi = self.slide_io.read_block(size=(self.opt.width // self.opt.downscale, self.opt.height // self.opt.downscale))
        # wsi = self.wsi.get_slide(self.opt.target_spacing * self.opt.downscale)
        wsi = self.wsi.get_thumbnail(
            (
                self.opt.width // self.opt.downscale,
                self.opt.height // self.opt.downscale,
            )
        )
        tissue_mask = get_tissue_mask(wsi)
        self.tissue_mask = tissue_mask
        return tissue_mask

    def get_patch_area_list(self):
        ret = []

        downscaled_height, downscaled_width = self.tissue_mask.shape

        downscaled_size = self.opt.adjust_patch_size // self.opt.downscale
        downscaled_stride = downscaled_size
        for y_loc in range(0, downscaled_height - downscaled_size, downscaled_stride):
            for x_loc in range(
                0, (downscaled_width // 2) - downscaled_size, downscaled_stride
            ):
                patch_mask = self.tissue_mask[
                    y_loc : y_loc + downscaled_size, x_loc : x_loc + downscaled_size
                ]
                if (
                    np.count_nonzero(patch_mask) / (downscaled_size * downscaled_size)
                    > 0.1
                ):
                    ret.append([x_loc * self.opt.downscale, y_loc * self.opt.downscale])
        return ret

    def __iter__(self):
        for x_loc, y_loc in self.data:
            # center_x, center_y = x_loc + self.opt.adjust_patch_size, y_loc + self.opt.adjust_patch_size
            try:
                target_patch = np.array(
                    self.wsi.read_region(
                        (x_loc, y_loc),
                        0,
                        (self.opt.adjust_patch_size, self.opt.adjust_patch_size),
                    ).convert("RGB")
                )
            except Exception as e:
                target_patch = np.zeros(
                    (self.opt.adjust_patch_size, self.opt.adjust_patch_size, 3), dtype=np.uint8
                )

            fn = self.transform_fn(image=target_patch)
            target_patch = fn["image"]


            target_patch = target_patch / 255.0


            yield {"position": (x_loc, y_loc), "images": (target_patch)}

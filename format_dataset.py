import config
import class_info
from PIL import Image
import numpy as np
from skimage.util import view_as_windows
import math
import matplotlib.pyplot as plt
import time


def resize_dataset(im_filename, gt_filename):
    data = np.zeros((config.rsz_fact[0], config.rsz_fact[1], 3), dtype=np.uint8)
    for idx, image_name in enumerate(im_filename):
        im = Image.open(image_name).resize((config.rsz_fact[1], config.rsz_fact[0]), Image.ANTIALIAS)
        gt = Image.open(gt_filename[idx]).resize((config.rsz_fact[1], config.rsz_fact[0]), Image.NEAREST)
        gt_rz = np.array(gt)
        for row in range(0, config.rsz_fact[0] - 1):
            for col in range(0, config.rsz_fact[1] - 1):
                data[row, col] = np.array(class_info.cls_clr[gt_rz[row, col]])
        gt_clr = Image.fromarray(data, 'RGB')

        # Save Images
        im.save(config.images_DB + '\\Image' + str(idx) + '.png')
        gt.save(config.gt_DB + '\\GT' + str(idx) + '.png')
        gt_clr.save(config.labels_DB + '\\Label' + str(idx) + '.png')

        print(idx, ' out of ', len(im_filename), ' converted...')


def generate_patches(im_filename, gt_filename):
    count = 1
    im_count = 0
    for iVal, imFile in enumerate(im_filename):
        # Load an image and extract patches
        im = Image.open(imFile)
        im_z = np.array(im)
        window_size = (config.patch_size, config.patch_size, config.channels)
        im_patches = view_as_windows(im_z, window_size, step=config.stride+(1,)).squeeze()
        print(im_patches.shape)

        # Load corresponding ground truth and extract patches
        gt = Image.open(gt_filename[iVal]).convert("L")
        gt_z = np.array(gt)
        window_size = (config.patch_size, config.patch_size)
        gt_patches = view_as_windows(gt_z, window_size, step=config.stride)
        print(gt_patches.shape)
        gt_patches_label = np.arange(gt_patches.shape[0] * gt_patches.shape[1]).reshape(gt_patches.shape[0],
                                                                                        gt_patches.shape[1])
        # Save extracted patches
        mid_pixel = math.floor(config.patch_size / 2)

        for row in range(0, gt_patches.shape[0]):
            for col in range(0, gt_patches.shape[1]):
                gt_patches_label[row][col] = gt_patches[row][col][mid_pixel][mid_pixel]
                temp_array = im_patches[row][col]
                im_ = Image.fromarray(temp_array.squeeze())
                patch_label = gt_patches[row][col][mid_pixel][mid_pixel]
                label_list = config.extract_classes['Label']
                nPos = findItem(label_list, patch_label)
                if nPos == []:
                    continue

                sv_path = config.out_folderDB + '\\' + str(nPos[0]) + config.extract_classes['Names'][nPos[0][0]] + \
                          '\\' + str(im_count).zfill(7) + '.png'
                im_.save(sv_path)
                im_count += 1
        print(str(round(count * 100 / (len(im_filename)), 2)) + '% completed')
        count += 1


def findItem(theList, item):
    return [[ind, theList[ind].index(item)] for ind in range(len(theList)) if item in theList[ind]]

"""
1. Resize the images and corresponding labels and store them
2. Generate pickled files using 'get_filenames_aspickled.py' to gather paths of
images and ground truth for training
3. Load path to images and ground truth from the pickled files
4. Generate patches of different sizes

"""

import pickle
import config
import format_dataset
import os

# ----------------------------------------------------------------------------------------
#   Input Configurations
# ----------------------------------------------------------------------------------------

# Stored in 'config.py'

# ----------------------------------------------------------------------------------------
#   Load Dataset (file paths)
# ----------------------------------------------------------------------------------------
with open(config.im_fileDB, 'rb') as fp:
    im_filename = pickle.load(fp)
with open(config.gt_fileDB, 'rb') as fp:
    gt_filename = pickle.load(fp)
# Sanity Check for correct data loading
assert(len(im_filename) == len(gt_filename))

# ----------------------------------------------------------------------------------------
#   Format Dataset (image files to nd arrays and resize)
# ----------------------------------------------------------------------------------------

if config.do_resize:
    format_dataset.resize_dataset(im_filename, gt_filename)
else:
    print("Skipping database formatting and resizing...")

# ----------------------------------------------------------------------------------------
#   Create Dataset (32x32, 64x64 and 128x128 patches)
# ----------------------------------------------------------------------------------------

# Generate Label folders in output dataset folder
# for i in range(0, config.num_classes):
#     fld = config.out_folderDB + '\\' + str(i)
#     if os.path.isdir(fld):
#         continue
#     else:
#         os.mkdir(fld)

for i in range(0, len(config.extract_classes['Names'])):
    fld = config.out_folderDB + '\\' + str(i) + '_' + config.extract_classes['Names'][i]
    if os.path.isdir(fld):
        continue
    else:
        os.mkdir(fld)


# Generate Patches of required sizes
format_dataset.generate_patches(im_filename, gt_filename)
print('Patch generation completed....')


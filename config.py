# File Structure
im_fileDB = r'C:\Users\palmaji\dataset\semantic\im_file_pickled.dat'
gt_fileDB = r'C:\Users\palmaji\dataset\semantic\gt_file_pickled.dat'
train_DB = r'C:\Users\palmaji\dataset\semantic\formatted_data'
images_DB = train_DB + '\\' + 'images'
gt_DB = train_DB + '\\' + 'gts'
labels_DB = train_DB + '\\' + 'labels'
out_folderDB = r'C:\Users\palmaji\dataset\semantic\patches_db_32x32'

# Parameters
do_resize = False
rsz_fact = [370, 1240]
num_classes = 29
npatch_pim = 3000
patch_size = 32     # can be 64 or 128 for square patches
stride = (4, 8)
channels = 3

# Class Conversion
extract_classes = {'Names': ['void', 'building', 'tree', 'sky', 'sidewalk',
                             'fence', 'road', 'grass', 'column', 'vehicle', 'person'],
                   'Label': [[0, 6, 27], [10, 11, 16], [8], [1], [3, 4], [15], [2, 5],
                             [7, 9], [12, 13, 14], [17, 18, 19, 20], [21, 22, 23, 25]]}

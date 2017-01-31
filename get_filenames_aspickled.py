import os
import pickle
import config
import format_dataset


# ----------------------------------------------------------------------------------------
#                       Input Configurations
# ----------------------------------------------------------------------------------------
imdir = r'C:\Users\palmaji\dataset\semantic\formatted_data\images'       # r'C:\Users\palmaji\dataset\semantic\imtrain'
gtdir = r'C:\Users\palmaji\dataset\semantic\formatted_data\gts'       # r'C:\Users\palmaji\dataset\semantic\gttrain'
im_fileDB = r'C:\Users\palmaji\dataset\semantic\formatted_data\im_file_pickled.dat'
gt_fileDB = r'C:\Users\palmaji\dataset\semantic\formatted_data\gt_file_pickled.dat'
imtype = ".png"
gttype = ".png"

# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
#                       Code starts here
# ----------------------------------------------------------------------------------------
im_filename = []
gt_filename = []

for file in os.listdir(imdir):
    if file.endswith(imtype):
        temp_filename = imdir + '\\' + file
        # print(temp_filename)
        im_filename.append(temp_filename)

for file in os.listdir(gtdir):
    if file.endswith(gttype):
        temp_filename = gtdir + '\\' + file
        # print(temp_filename)
        gt_filename.append(temp_filename)

with open(im_fileDB, 'wb') as fp:
    pickle.dump(im_filename, fp)
with open(gt_fileDB, 'wb') as fp:
    pickle.dump(gt_filename, fp)


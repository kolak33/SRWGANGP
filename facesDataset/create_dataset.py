from scipy.misc import imresize
from skimage import img_as_ubyte
import imageio
import os
import csv
from scipy import misc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cropped_dir', dest='cropped_dir', default='data_crop_128_png')
args = parser.parse_args()

# create low, medium and high resolution datasets used in training based on original HD dataset
dirs = ['faces/32/train_imgs/train', 'faces/32/val_imgs/val', 'faces/32/test_imgs/test',
       'faces/128/train_imgs/train', 'faces/128/val_imgs/val', 'faces/128/test_imgs/test',
       '../Weights', '../Images/HR', '../Images/SRWGANGP', '../Images/NN_128']
original_dataset_path = args.cropped_dir

HIGH_SCALE = 4.0
# create directories
for f in dirs:
    if not os.path.exists(f):
        os.makedirs(f)

original_dataset_dir = os.fsencode(original_dataset_path)

print('Creating datasets... This will take some time...')
with open('list_eval_partition.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    next(reader, None) # skip header
    for row in reader:
        filename = row[0]
        result_directory_type = int(row[1])

        filename = filename[0:-4] + '.png'
        img_path = os.path.join(original_dataset_path, filename)
        # create HD image
        hd_img = imageio.imread(img_path)  

        lr_img = misc.imresize(hd_img, 1.0 / HIGH_SCALE, interp="bicubic")

        hd_img = img_as_ubyte(hd_img)
        lr_img = img_as_ubyte(lr_img)

        
        if result_directory_type == 0: # train set
            img_dir_type = 'train_imgs/train'
        elif result_directory_type == 1: # validation set
            img_dir_type = 'val_imgs/val'
        else: # test set
            img_dir_type = 'test_imgs/test'
        
        hd_img_path = os.path.join('faces/128', img_dir_type, filename)
        lr_img_path = os.path.join('faces/32', img_dir_type, filename)
        
        imageio.imwrite(hd_img_path, hd_img)
        imageio.imwrite(lr_img_path, lr_img)
csvFile.close()    
    
print('Complete.')    
    
    
    
    
    
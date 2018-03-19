# -*- coding: utf-8 -*-

import glob
import os
import cv2
import numpy as np
from multiprocessing import Pool

patch_size, stride = 40, 10
aug_times = 1

def data_aug(img, mode=0):
    
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))
    
def gen_patches(file_name):

    # read image
    img = cv2.imread(file_name, 0)  # gray scale
    h, w = img.shape
    scales = [1, 0.9, 0.8, 0.7]
    patches = []

    for s in scales:
        h_scaled, w_scaled = int(h*s),int(w*s)
        img_scaled = cv2.resize(img, (h_scaled,w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img_scaled[i:i+patch_size, j:j+patch_size]
                # data aug
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0,8))
                    patches.append(x_aug)
    
    return patches

if __name__ == '__main__':
    # parameters
    src_dir = './data/Train400/'
    save_dir = './data/npy_data/'
    file_list = glob.glob(src_dir+'*.png')  # get name list of all .png files
    num_threads = 16    
    print('Start...')
    # initrialize
    res = []
    # generate patches
    for i in range(0,len(file_list),num_threads):
        # use multi-process to speed up
        p = Pool(num_threads)
        patch = p.map(gen_patches,file_list[i:min(i+num_threads,len(file_list))])
        #patch = p.map(gen_patches,file_list[i:i+num_threads])
        for x in patch:
            res += x
        
        print('Picture '+str(i)+' to '+str(i+num_threads)+' are finished...')
    
    # save to .npy
    res = np.array(res, dtype='uint8')
    print('Shape of result = ' + str(res.shape))
    print('Saving data...')
    if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    np.save(save_dir+'clean_patches.npy', res)
    print('Done.')       
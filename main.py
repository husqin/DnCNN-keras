# -*- coding: utf-8 -*-

import argparse
import logging
import os, time, glob
import PIL.Image as Image
import numpy as np
import pandas as pd
#from keras import backend as K
#import tensorflow as tf
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
from keras.optimizers import Adam
from skimage.measure import compare_psnr, compare_ssim
import models
#from util import *

## Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='DnCNN', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--train_data', default='./data/npy_data/clean_patches.npy', type=str, help='path of train data')
parser.add_argument('--test_dir', default='./data/Test/Set68', type=str, help='directory of test dataset')
parser.add_argument('--sigma', default=25, type=int, help='noise level')
parser.add_argument('--epoch', default=50, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=5, type=int, help='save model at every x epoches')
parser.add_argument('--pretrain', default=None, type=str, help='path of pre-trained model')
parser.add_argument('--only_test', default=False, type=bool, help='train and test or only test')
args = parser.parse_args()

if not args.only_test:
    save_dir = './snapshot/save_'+ args.model + '_' + 'sigma' + str(args.sigma) + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # log
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%Y %H:%M:%S',
                    filename=save_dir+'info.log',
                    filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-6s: %(levelname)-6s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    logging.info(args)
    
else:
    save_dir = '/'.join(args.pretrain.split('/')[:-1]) + '/'

      
def load_train_data():
    
    logging.info('loading train data...')   
    data = np.load(args.train_data)
    logging.info('Size of train data: ({}, {}, {})'.format(data.shape[0],data.shape[1],data.shape[2]))
    
    return data

def step_decay(epoch):
    
    initial_lr = args.lr
    if epoch<50:
        lr = initial_lr
    else:
        lr = initial_lr/10
    
    return lr

def train_datagen(y_, batch_size=8):
    
    # y_ is the tensor of clean patches
    indices = list(range(y_.shape[0]))
    while(True):
        np.random.shuffle(indices)    # shuffle
        for i in range(0, len(indices), batch_size):
            ge_batch_y = y_[indices[i:i+batch_size]]
            noise =  np.random.normal(0, args.sigma/255.0, ge_batch_y.shape)    # noise
            #noise =  K.random_normal(ge_batch_y.shape, mean=0, stddev=args.sigma/255.0)
            ge_batch_x = ge_batch_y + noise  # input image = clean image + noise
            yield ge_batch_x, ge_batch_y
        
def train():
    
    data = load_train_data()
    data = data.reshape((data.shape[0],data.shape[1],data.shape[2],1))
    data = data.astype('float32')/255.0
    # model selection
    if args.pretrain:   model = load_model(args.pretrain, compile=False)
    else:   
        if args.model == 'DnCNN': model = models.DnCNN()
    # compile the model
    model.compile(optimizer=Adam(), loss=['mse'])
    
    # use call back functions
    ckpt = ModelCheckpoint(save_dir+'/model_{epoch:02d}.h5', monitor='val_loss', 
                    verbose=0, period=args.save_every)
    csv_logger = CSVLogger(save_dir+'/log.csv', append=True, separator=',')
    lr = LearningRateScheduler(step_decay)
    # train 
    history = model.fit_generator(train_datagen(data, batch_size=args.batch_size),
                    steps_per_epoch=len(data)//args.batch_size, epochs=args.epoch, verbose=1, 
                    callbacks=[ckpt, csv_logger, lr])
    
    return model

def test(model):
    
    print('Start to test on {}'.format(args.test_dir))
    out_dir = save_dir + args.test_dir.split('/')[-1] + '/'
    if not os.path.exists(out_dir):
            os.mkdir(out_dir)
            
    name = []
    psnr = []
    ssim = []
    file_list = glob.glob('{}/*.png'.format(args.test_dir))
    for file in file_list:
        # read image
        img_clean = np.array(Image.open(file), dtype='float32') / 255.0
        img_test = img_clean + np.random.normal(0, args.sigma/255.0, img_clean.shape)
        img_test = img_test.astype('float32')
        # predict
        x_test = img_test.reshape(1, img_test.shape[0], img_test.shape[1], 1) 
        y_predict = model.predict(x_test)
        # calculate numeric metrics
        img_out = y_predict.reshape(img_clean.shape)
        img_out = np.clip(img_out, 0, 1)
        psnr_noise, psnr_denoised = compare_psnr(img_clean, img_test), compare_psnr(img_clean, img_out)
        ssim_noise, ssim_denoised = compare_ssim(img_clean, img_test), compare_ssim(img_clean, img_out)
        psnr.append(psnr_denoised)
        ssim.append(ssim_denoised)
        # save images
        filename = file.split('/')[-1].split('.')[0]    # get the name of image file
        name.append(filename)
        img_test = Image.fromarray((img_test*255).astype('uint8'))
        img_test.save(out_dir+filename+'_sigma'+'{}_psnr{:.2f}.png'.format(args.sigma, psnr_noise))
        img_out = Image.fromarray((img_out*255).astype('uint8')) 
        img_out.save(out_dir+filename+'_psnr{:.2f}.png'.format(psnr_denoised))
    
    psnr_avg = sum(psnr)/len(psnr)
    ssim_avg = sum(ssim)/len(ssim)
    name.append('Average')
    psnr.append(psnr_avg)
    ssim.append(ssim_avg)
    print('Average PSNR = {0:.2f}, SSIM = {1:.2f}'.format(psnr_avg, ssim_avg))
    
    pd.DataFrame({'name':np.array(name), 'psnr':np.array(psnr), 'ssim':np.array(ssim)}).to_csv(out_dir+'/metrics.csv', index=True)
    
if __name__ == '__main__':   
    
    if args.only_test:
        model = load_model(args.pretrain, compile=False)
        test(model)
    else:
        model = train()
        test(model)       
    

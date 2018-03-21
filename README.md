# DnCNN-keras     
A keras implemention of the paper [Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](http://www4.comp.polyu.edu.hk/~cslzhang/paper/DnCNN.pdf)

### Dependence
```
tensorflow
keras2
numpy
opencv
```

### Prepare train data
```
$ python data.py
```

Clean patches are extracted from 'data/Train400' and saved in 'data/npy_data'.
### Train
```
$ python main.py
```

Trained models are saved in 'snapshot'.
### Test
```
$ python main.py --only_test True --pretrain 'path of saved model'
```

Noisy and denoised images are saved in 'snapshot'.

### Results

#### Gaussian Denoising

The average PSNR(dB) results of different methods on the BSD68 dataset.

|  Noise Level | BM3D | DnCNN-S | DnCNN-keras |
|:-------:|:-------:|:-------:|:-------:|
| 25  |  28.57 | 29.23 | 29.21  |








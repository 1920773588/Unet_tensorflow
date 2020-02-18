import imageUnits
import tensorflow as tf
import numpy as tf
import Unet

train_path = 'data/val/*.tif'
batch_size = 1
channels = 3
nclass = 2
pad = True

predition_data_provider = imageUnits.ImageProvider(path=train_path, bathsize=batch_size, shuffle_data=True,
                                               channels=channels, nclass=nclass, start=0, pad=pad)
unet = Unet.unet(predition_data_provider, layers=3, first_feature_num=64, conv_size=3, pool_size=2, batch_size=batch_size,
                 channels=channels, nclass=nclass, save_path='layer3', white_channel_weight=0.4, pad=pad)
unet.predite()
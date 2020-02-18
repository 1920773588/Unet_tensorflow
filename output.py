import Unet
import imageUnits
train_path = 'data/train/*.tif'
batch_size = 50
channels = 3
nclass = 2

train_data_provider = imageUnits.ImageProvider(path=train_path, bathsize=batch_size, shuffle_data=True,
                                               channels=channels, nclass=nclass, pad=True)
unet = Unet.unet(train_data_provider, layers=3, first_feature_num=32, conv_size=3, pool_size=2, batch_size=batch_size,
                 channels=channels, nclass=nclass, save_path='layer3', white_channel_weight=0.4, pad=True)
unet.output()


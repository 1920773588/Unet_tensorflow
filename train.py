import imageUnits
import Unet
import os

train_path = 'data/train/*.tif'
batch_size = 32
channels = 3
nclass = 2
pad = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_data_provider = imageUnits.ImageProvider(path=train_path, bathsize=batch_size, shuffle_data=True,
                                               channels=channels, nclass=nclass, pad=pad)
unet = Unet.unet(train_data_provider, layers=3, first_feature_num=32, conv_size=3, pool_size=2, batch_size=batch_size,
                 channels=channels, nclass=nclass, save_path='layer3', white_channel_weight=0.4, pad=pad)
best_acc, epoch = unet.trian(epochs=50, train_iters=50, keep_prob=0.8, learn_rate=0.0005, restore=True, save_steps=25, loss_name='cross')
print(best_acc, epoch)
#unet.black_test()
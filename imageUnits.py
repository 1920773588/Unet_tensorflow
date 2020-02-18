import tensorflow as tf
import numpy as np
import glob
from PIL import Image
pad_size = 104
padding = int((pad_size-64)/2)

class ImageProvider:
    def __init__(self, path='data/train/*.tif', bathsize=1, shuffle_data=True, channels=3, nclass=2, start=0, pad=False):
        self.channels = channels
        self.nclass = nclass
        self.datas_path = self.find_data_path(path)  # 加载图片路径
        self.bathsize = bathsize
        self.bathnum = len(self.datas_path) // bathsize
        self.shuffle_data = shuffle_data
        self.pad = pad
        if self.shuffle_data:
            self.shuffle()
        else:
            self.labels_path = [i[:-4] + "_mask.tif" for i in self.datas_path]  # 获得对应label路径
        self.count = start

        self.data_shape = np.array(Image.open(self.datas_path[0]), dtype=float).shape
        self.label_shape = np.array(Image.open(self.labels_path[0]), dtype=float).shape
        if len(self.data_shape) == 2:
            self.channels = 1
        else:
            self.channels = self.data_shape[-1]
        if len(self.label_shape) == 2:
            self.nclass = 2
        else:
            self.nclass = self.label_shape[-1]

    def find_data_path(self, path):
        images_path = glob.glob(path)
        datas_path = [i for i in images_path if '_mask.tif' not in i]
        return datas_path

    def shuffle(self):
        np.random.shuffle(self.datas_path)
        self.labels_path = [i[:-4] + "_mask.tif" for i in self.datas_path]

    def open_data_image(self, path):
        """
        :param path：path of data image
        :return np.array of image after normalizing
        """
        data = np.array(Image.open(path), np.float)
        data = np.fabs(data)
        data -= np.amin(data)
        if np.amax(data) != 0:
            data /= np.amax(data)
        if self.pad:
            data = np.pad(data, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
            return data.reshape(1, pad_size, pad_size, self.channels)
        return data.reshape(1, self.data_shape[0], self.data_shape[1], self.channels)

    def open_label_image(self, path):
        """
        :param
            path：path of label image
        :return:
        """
        label = np.array(Image.open(path), np.bool)
        labels = np.zeros((label.shape[0], label.shape[1], self.nclass), dtype=np.float)
        # for i in range(label.shape[0]):
        #     for j in range(label.shape[1]):
        #         if label[i][j] == 0:
        #             labels[i][j][0] = 0
        #             labels[i][j][1] = 1
        #         else:
        #             labels[i][j][0] = 1
        #             labels[i][j][1] = 0
        labels[..., 0] = label
        labels[..., 1] = ~label
        # label1 = np.array(Image.open(path), np.float)
        # save_image(img=label1, path='mask_ini.jpg')
        # create_save_img(img=labels.reshape(1, self.label_shape[0], self.label_shape[1], self.nclass), path='mask_1.jpg')
        return labels.reshape(1, self.label_shape[0], self.label_shape[1], self.nclass)

    def next_batch(self):
        """
        :returns:  a batch of datas and labels
        """
        if self.bathsize == 1:
            self.count += 1
            return self.open_data_image(self.datas_path[self.count - 1]), \
                   self.open_label_image(self.labels_path[self.count - 1])
        if self.pad:
            datas = np.zeros((self.bathsize, pad_size, pad_size, self.channels), dtype=np.float)
        else:
            datas = np.zeros((self.bathsize, self.data_shape[0], self.data_shape[1], self.channels), dtype=np.float)
        labels = np.zeros((self.bathsize, self.label_shape[0], self.label_shape[1], self.nclass), dtype=np.float)
        for i in range(self.bathsize):
            if self.count == len(self.datas_path):
                self.count = 0
            datas[i] = self.open_data_image(self.datas_path[self.count])
            labels[i] = self.open_label_image(self.labels_path[self.count])
            self.count += 1

        return datas, labels


def save_image(img, path, format='JPG'):
    """
    :param
        img: np.array of the image to be saved
        path：save path of the image
        format: format of image to be saved
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        Image.fromarray(img.astype('uint8'), 'RGB').save(path, format=format)
    else:
        Image.fromarray(img.astype('uint8')).save(path)


def cut_image(img, shape):
    """
    cut the array(img) to the given shape by removing the border
    :param img:[batch_size, nx, ny, channels]
    :param shape:[batch_size, nx1, ny1, channels]
    :return:a new np.array after cut [batch_size, nx1, ny1, channels]
    """
    diff_axis0 = img.shape[1] - shape[1]
    diff_axis1 = img.shape[2] - shape[2]

    axis0_left = int(diff_axis0 // 2)
    axis0_right = int(axis0_left + shape[1])
    axis1_left = int(diff_axis1 // 2)
    axis1_right = int(axis1_left + shape[2])

    data = img[:, axis0_left:axis0_right, axis1_left:axis1_right, :]
    return data


def pixel_softmax(output):
    max_class = tf.reduce_max(output, axis=3, keep_dims=True)
    exp = tf.exp(output - max_class)
    exp = tf.exp(output)
    sum = tf.reduce_sum(exp, axis=3, keep_dims=True)
    return exp / sum


def create_save_img(img, path):
    num = img.shape[0]
    for i in range(num):
        picture = img[i, :, :, 0]
        picture1 = img[i, :, :, 1]
        # picture = np.zeros((img.shape[1], img.shape[2]))
        # for j in range(img.shape[1]):
        #     for z in range(img.shape[2]):
        #         #logits channel0 白 channel1 黑
        #         if img[i, j, z, 0] > img[i, j, z, 1]:
        #             picture[j][z] = 255 #白
        #         else:
        #             picture[j][z] = 0 #
        # np.savetxt('out_txt/0.txt', img[0, :, :, 0])
        # np.savetxt('out_txt/1.txt', img[0, :, :, 1])
        picture[picture >= 0.5] = 255
        picture[picture < 1] = 0
        #np.savetxt('out_txt/picture.txt', picture)
        save_image(img=picture, path=path, format=path.split('.')[-1])
    return
#
# train_data_provider = ImageProvider(path='data/train/*.tif', bathsize=10, shuffle_data=True)
# datas, labels = train_data_provider.next_batch()
# print(datas.shape)
# print(labels.shape)

# train_data_path = 'data/train/*.tif'
# test_data_path = 'data/test-image/*.tif'
# #返回所有图片目录
# images = glob.glob(train_data_path)
# datas = [i for i in images if '_mask.tif' not in i]
# labels = [i for i in images if '_mask.tif' in i]
#
# image1 = np.array(Image.open(labels[100]), np.float)
# print(image1)
# im = Image.fromarray(image1.astype('uint8'))
# #save_image(img=image, path='test.jpg', type='RGB', mode='jpg')
# im.show()
# print(len(datas))
# print(len(labels))
# print(image.shape)

import tensorflow as tf
import numpy as np
import imageUnits
import os
import glob
from PIL import Image

pad_size = 104
padding = int((pad_size - 64) / 2)


def weight_variable(shape, stddev=0.2):
    shape = tf.cast(shape, tf.int32)
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name='weight')


def weight_variable_deconv(shape, stddev=0.2):
    shape = tf.cast(shape, tf.int32)
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name='deconv_weight')


def bias_variable(shape, name, init=0.1):
    return tf.Variable(tf.constant(init, shape=shape), name=name)


def conv(x, w, b, keep_prob, pad):
    with tf.name_scope('conv'):
        if pad:
            conv_and_bias = tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding='VALID') + b
        else:
            conv_and_bias = tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding='SAME') + b
        return tf.nn.dropout(conv_and_bias, keep_prob=keep_prob)


def deconv(x, w, b, stride, pad):
    with tf.name_scope('deconv'):
        x_shape = tf.shape(x)
        out_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] // 2])
        if pad:
            return tf.nn.conv2d_transpose(x, w, out_shape, strides=[1, stride, stride, 1], padding='VALID',
                                          name='deconv') + b
        else:
            return tf.nn.conv2d_transpose(x, w, out_shape, strides=[1, stride, stride, 1], padding='SAME',
                                          name='deconv') + b


def max_pool(x, size):
    with tf.name_scope('max_pool'):
        return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='VALID', name='max_pool')


def concat(conv, deconv):
    with tf.name_scope('concat'):
        conv_shape = tf.shape(conv)
        deconv_shape = tf.shape(deconv)
        offsets = [0, (conv_shape[1] - deconv_shape[1]) // 2, (conv_shape[2] - deconv_shape[2]) // 2, 0]
        conv_slice = tf.slice(conv, offsets, [-1, deconv_shape[1], deconv_shape[2], -1])
        return tf.concat([conv_slice, deconv], axis=3)


class unet():
    def __init__(self, data_provider, layers, first_feature_num, conv_size, pool_size, batch_size,
                 channels=3, nclass=2, save_path='train_out', white_channel_weight=1, pad=False):
        self.data_provider = data_provider
        self.layers = layers
        self.channels = channels
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.nclass = nclass
        self.batch_size = batch_size
        self.first_feature_num = first_feature_num
        self.conv_size = conv_size
        self.pool_size = pool_size
        self.save_path = save_path
        self.white_weight = white_channel_weight
        self.pad = pad

        self.logits, self.labels, self.variables, self.output_size = self.create_net([None, None, None, channels],
                                                                                     [None, None, None, nclass],
                                                                                     self.pad)
        self.prediction = imageUnits.pixel_softmax(self.logits)
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.prediction, 3), tf.argmax(self.labels, 3)), tf.float32))

    def create_net(self, shape1, shape2, pad):
        """
        create Unet
        :param shape1: input tensor shape [batch_size, nx, ny, channels]
        :param shape2: output tensor shape [batch_size, nx1, ny1, nclass]
        :return: logits[batch_size, nx1, ny1, nclass],
                labels[batch_size, nx1, ny1, nclass],
                a list of all variales used in Unet,
                size:nx1(ny1)
        """
        self.x = tf.placeholder('float', shape1, name='input')
        self.y = tf.placeholder('float', shape2, name='label')
        # 将np数组转化为tensor
        node = tf.reshape(self.x, tf.stack([-1, tf.shape(self.x)[1], tf.shape(self.x)[2], self.channels]))
        labels = tf.reshape(self.y, tf.stack([-1, tf.shape(self.y)[1], tf.shape(self.y)[2], self.nclass]))

        down_layers = [i for i in range(self.layers)]
        Variables = []
        size = 64
        if pad:
            size = pad_size
        # print(labels.dtype)
        for i in range(self.layers):
            with tf.name_scope("down_conv_" + str(i)):
                features = 2 ** i * self.first_feature_num
                if i == 0:
                    w1 = weight_variable([self.conv_size, self.conv_size, self.channels, features], stddev=0.1)
                else:
                    w1 = weight_variable([self.conv_size, self.conv_size, features // 2, features], stddev=0.1)

                w2 = weight_variable([self.conv_size, self.conv_size, features, features], stddev=0.1)
                b1 = bias_variable([features], init=0.1, name='bias')
                b2 = bias_variable([features], init=0.1, name='bias')
                conv1 = conv(node, w1, b1, keep_prob=self.keep_prob, pad=pad)
                relu1 = tf.nn.relu(conv1)
                conv2 = conv(relu1, w2, b2, keep_prob=self.keep_prob, pad=pad)
                relu2 = tf.nn.relu(conv2)

                down_layers[i] = relu2
                size -= 2 * 2 * (self.conv_size // 2)
                Variables.append(w1)
                Variables.append(w2)
                Variables.append(b1)
                Variables.append(b2)
                if i < self.layers - 1:
                    node = max_pool(relu2, size=self.pool_size)
                    size /= self.pool_size

        node = relu2
        print('down_size' + str(size))
        for i in range(self.layers - 2, -1, -1):
            with tf.name_scope('up_conv_' + str(i)):
                features = 2 ** i * self.first_feature_num
                w_deconv = weight_variable_deconv([self.pool_size, self.pool_size, features, features * 2], stddev=0.1)
                b_deconv = bias_variable([features], name='deconv_bias')
                deconv1 = tf.nn.relu(deconv(node, w_deconv, b_deconv, stride=self.pool_size, pad=pad))

                x_concat = concat(down_layers[i], deconv1)

                w1 = weight_variable([self.conv_size, self.conv_size, features * 2, features], stddev=0.1)
                w2 = weight_variable([self.conv_size, self.conv_size, features, features], stddev=0.1)
                b1 = bias_variable([features], init=0.1, name='bais_up')
                b2 = bias_variable([features], init=0.1, name='bais_up')

                conv1 = conv(x_concat, w1, b1, keep_prob=self.keep_prob, pad=pad)
                relu1 = tf.nn.relu(conv1)
                conv2 = conv(relu1, w2, b2, keep_prob=self.keep_prob, pad=pad)
                relu2 = tf.nn.relu(conv2)

                node = relu2
                size *= self.pool_size
                size -= 2 * 2 * (self.conv_size // 2)
                Variables.append(w_deconv)
                Variables.append(b_deconv)
                Variables.append(w1)
                Variables.append(w2)
                Variables.append(b1)
                Variables.append(b2)

        with tf.name_scope('output'):
            w = weight_variable([1, 1, self.first_feature_num, self.nclass], stddev=0.1)
            b = bias_variable([self.nclass], init=0.1, name='bais_output')
            logits = conv(node, w, b, keep_prob=1, pad=pad)
            Variables.append(w)
            Variables.append(b)

        if not pad:
            size = 64
        else:
            offsets = [0, int((size - 64) // 2), int((size - 64) // 2), 0]
            logits = tf.slice(logits, offsets, [-1, 64, 64, -1])
            size = 64

        print('Created Unet: layers:' + str(self.layers), 'output_size:' + str(size),
              'batch_size:' + str(self.batch_size))
        return logits, labels, Variables, size

    def create_optimizer(self, global_step, learn_rate, loss_name, decay_steps=50, decay_rate=0.95):
        """
        create optimizer tensor
        :param global_step: total number of training
        :param learn_rate: the started learning rate
        :param decay_steps:
        :param decay_rate:
        :return:optimizer tensor and loss tensor
        """
        loss = self.compute_loss(self.logits, self.labels, loss_name)

        with tf.name_scope('learning_rate'):
            self.learn_rate_node = tf.train.exponential_decay(learn_rate, global_step, decay_steps, decay_rate)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss, global_step=global_step)
            # optimizer = tf.train.MomentumOptimizer(learning_rate=self.learn_rate_node, momentum=0.3).minimize(loss, global_step)
        return optimizer, loss

    def compute_loss(self, logits, labels, loss_name):
        """
        compute total loss tensor
        :param logits: logits tensor[batch_size, x, y, nclass]
        :param labels: label tensor[batch_size, x, y, nclass]
        :return: loss tensor
        """
        if loss_name == 'cross':
            print('Cross_entropy_loss')
            with tf.name_scope('cross_entropy_loss'):
                logits = tf.reshape(logits, [-1, self.nclass])
                labels = tf.reshape(labels, [-1, self.nclass])
                # weigths = tf.constant([self.white_weight, 0.1], tf.float32, [self.nclass, 1], name='channel_weight')
                # weigths_map = tf.matmul(labels, weigths)
                # weigths_map = tf.reduce_sum(weigths_map, axis=1)
                loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)

                # weighted_loss = tf.multiply(loss_map, weigths_map)
                # loss = tf.reduce_mean(weighted_loss)
                loss = tf.reduce_mean(loss_map)
        elif loss_name == 'dice':
            print('Dice_loss')
            with tf.name_scope('loss_dice'):
                eps = 1e-5
                prediction = imageUnits.pixel_softmax(logits)
                intersection = tf.reduce_sum(prediction * labels)
                union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(labels)
                loss = -(2 * intersection / (union))
        elif loss_name == 'focal':
            print('focal_loss')
            with tf.name_scope('focal_loss'):
                gamma = 2
                alpha = 0.85
                prediction = imageUnits.pixel_softmax(logits)
                pt_1 = tf.where(tf.equal(labels, 1), prediction, tf.ones_like(prediction))
                pt_0 = tf.where(tf.equal(labels, 0), prediction, tf.zeros_like(prediction))
                pt_1 = tf.clip_by_value(pt_1, 1e-9, .999)
                pt_0 = tf.clip_by_value(pt_0, 1e-9, .999)
                loss = -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.log(pt_1)) - tf.reduce_sum(
                    (1 - alpha) * tf.pow(pt_0, gamma) * tf.log(1. - pt_0))
        else:
            print('None loss')
        # self.k_means = self.k_means()
        # loss += 0.001 * self.k_means
        # L2
        # L2 = 0.00001 * sum([tf.nn.l2_loss(i) for i in self.variables])
        # loss += L2
        return loss

    def k_means(self):
        aaa = tf.reshape(self.prediction[:, :, :, 0], [self.batch_size, self.output_size, self.output_size])
        one = tf.ones_like(aaa)
        zero = tf.zeros_like(aaa)
        aaa = tf.where(aaa > 0.5, x=one, y=zero)

        x = np.arange(self.output_size) + 1
        x_mask = np.tile(x, (self.batch_size, self.output_size, 1))
        y = x_mask[0, :, :].T
        y_mask = np.tile(y, (self.batch_size, 1, 1))

        x_mask = tf.Variable(x_mask, dtype=tf.float32)
        y_mask = tf.Variable(y_mask, dtype=tf.float32)
        one_count = tf.cast(tf.count_nonzero(aaa, axis=[1, 2]) + 1, dtype=tf.float32)
        ccc = tf.multiply(aaa, x_mask)
        ccc_y = tf.multiply(aaa, y_mask)
        self.aaa = aaa
        self.ccc = ccc
        x_center = tf.divide(tf.reduce_sum(ccc, axis=[1, 2]), one_count)
        y_center = tf.divide(tf.reduce_sum(ccc_y, axis=[1, 2]), one_count)
        self.sum = tf.reduce_sum(ccc, axis=[1, 2])
        x_add = tf.zeros(shape=[1, self.output_size, self.output_size])
        y_add = tf.zeros(shape=[1, self.output_size, self.output_size])
        self.txt = x_center
        for i in range(self.batch_size):
            a = tf.tile(tf.reshape(x_center[i], [1, 1]), [self.output_size, self.output_size])
            b = tf.tile(tf.reshape(y_center[i], [1, 1]), [self.output_size, self.output_size])

            new_x = tf.reshape(tf.where(ccc[i] == 0, a, ccc[i]), shape=[1, self.output_size, self.output_size])
            x_add = tf.concat([x_add, new_x], axis=0)
            new_y = tf.reshape(tf.where(ccc_y[i] == 0, b, ccc_y[i]), shape=[1, self.output_size, self.output_size])
            y_add = tf.concat([y_add, new_y], axis=0)
        x_add = x_add[1:, :, :]
        y_add = y_add[1:, :, :]
        eee = tf.reshape([tf.tile(tf.reshape(x_center[i], [1, 1]), (self.output_size, self.output_size)) for i in
                          range(self.batch_size)], shape=[self.batch_size, self.output_size, self.output_size])
        eee_y = tf.reshape([tf.tile(tf.reshape(y_center[i], [1, 1]), (self.output_size, self.output_size)) for i in
                            range(self.batch_size)], shape=[self.batch_size, self.output_size, self.output_size])

        loss = tf.reduce_mean(
            tf.reduce_sum(tf.add(tf.square(x_add - eee), tf.square(y_add - eee_y)), axis=[1, 2]) / one_count)
        return loss

    def predite(self):
        init = tf.global_variables_initializer()
        with tf.name_scope('predition'):
            with tf.Session() as sess:
                sess.run(init)
                self.sess_restore(sess)
                sum_acc = 0
                for i in range(100):
                    batch_x, batch_y = self.data_provider.next_batch()

                    predition, label, logits, acc = sess.run((self.prediction, self.labels, self.logits, self.accuracy),
                                                             feed_dict={self.x: batch_x,
                                                                        self.y: imageUnits.cut_image(batch_y, [None,
                                                                                                               self.output_size,
                                                                                                               self.output_size,
                                                                                                               None]),
                                                                        self.keep_prob: 1.}
                                                             )
                    label = np.array(label, dtype=np.float)
                    imageUnits.create_save_img(label, 'output/mask/mask' + str(i) + '.jpg')
                    imageUnits.create_save_img(predition, 'output/predition/predition' + str(i) + '.jpg')
                    sum_acc += acc
                print(str(sum_acc / 100))
                # np.savetxt('out_txt/logits_0.txt', logits[0, :, :, 0])
                # np.savetxt('out_txt/logits_1.txt', logits[0, :, :, 1])
        return

    def output(self):
        test_data_path = 'data/test-images/*.tif'
        images = glob.glob(test_data_path)
        images = [i for i in images if '_mask.tif' not in i]
        init = tf.global_variables_initializer()
        sum_acc = 0
        with tf.name_scope('output'):
            with tf.Session() as sess:
                sess.run(init)
                self.sess_restore(sess)
                for i in range(len(images)):
                    path = images[i]
                    name = path[17:]
                    data = self.data_provider.open_data_image(path)
                    label = self.data_provider.open_label_image(path[0:-4] + '_mask.tif')
                    # if self.pad:
                    #     data = np.pad(data, ((0, 0), (padding, padding), (padding, padding), (0, 0)), mode='constant')
                    predition, acc = sess.run((self.prediction, self.accuracy),
                                              feed_dict={self.x: data,
                                                         self.y: label,
                                                         self.keep_prob: 1.}
                                              )
                    imageUnits.create_save_img(predition, 'output/output/' + name[0:-4] + '_mask.tif')
                    sum_acc += acc
                print(sum_acc / len(images))
        return

    def test(self, sess, name):
        data = self.data_provider.open_data_image('data\\train\TCGA_CS_4941_19960909_14.tif')
        label = self.data_provider.open_label_image('data\\train\TCGA_CS_4941_19960909_14_mask.tif')

        predition, accuracy = sess.run((self.prediction, self.accuracy),
                                       feed_dict={self.x: data,
                                                  self.y: imageUnits.cut_image(label, [None, self.output_size,
                                                                                       self.output_size, None]),
                                                  self.keep_prob: 1.})
        imageUnits.create_save_img(predition, 'test/' + name + '_acc' + str(accuracy) + '.jpg')
        # np.savetxt('out_txt/predition_0.txt', predition[0, :, :, 0])
        # np.savetxt('out_txt/predition_1.txt', predition[0, :, :, 1])
        return

    def verify(self, sess):
        test_data_path = 'data/test-images/*.tif'
        images = glob.glob(test_data_path)
        images = [i for i in images if '_mask.tif' not in i]

        sum_acc = 0
        for i in range(len(images)):
            path = images[i]
            data = self.data_provider.open_data_image(path)
            label = self.data_provider.open_label_image(path[0:-4] + '_mask.tif')
            acc = sess.run(self.accuracy,
                           feed_dict={self.x: data,
                                      self.y: label,
                                      self.keep_prob: 1.}
                           )
            sum_acc += acc
        return sum_acc / len(images)
        # dataprovider = imageUnits.ImageProvider(path='data/test-images/*.tif', bathsize=1, shuffle_data=False, channels=2,
        #                                         nclass=3, pad=self.pad)
        # sum_acc = 0
        # for i in range(77):
        #     batch_x, batch_y = dataprovider.next_batch()
        #     accuracy = sess.run(self.accuracy,
        #                         feed_dict={self.x: batch_x,
        #                                    self.y: batch_y,
        #                                    self.keep_prob: 1.})
        #     sum_acc += accuracy
        # return sum_acc / 77

    def save_sess(self, sess, name):
        """
        save a session
        :param sess: session to save
        :param name: save name
        :return: save path
        """
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        saver = tf.train.Saver()
        return saver.save(sess, os.path.join(self.save_path, name))

    def sess_restore(self, sess):
        """
        restore a session
        :param sess:current session
        :param name:model saved name
        :return:
        """
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.save_path)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

    def black_test(self):
        dataprovider = imageUnits.ImageProvider(path='data/val/*.tif', bathsize=50, shuffle_data=False, channels=2,
                                                nclass=3, pad=False)
        sum_acc = 0
        for i in range(16):
            batch_x, batch_y = dataprovider.next_batch()
            zero = np.zeros((50, 64, 64, 1))
            one = np.ones((50, 64, 64, 1))
            batch_x = np.concatenate((zero, one), axis=3)
            accuracy = np.mean(
                np.equal(np.argmax(batch_x, 3), np.argmax(batch_y, 3)).astype(np.float))

            sum_acc += accuracy
        print(sum_acc / 16)

    def trian(self, epochs, train_iters, keep_prob, learn_rate=0.2, restore=False, save_steps=100, loss_name='cross'):
        """
        train the net had created
        :param epochs: number of epochs
        :param train_iters: number of training every epoch
        :param keep_prob: dropout probability tensor
        :param learn_rate: the started learning rate
        :return:
        """
        sum_steps = tf.Variable(epochs * train_iters, name="global_step")
        self.optimizer, self.loss = self.create_optimizer(global_step=sum_steps, learn_rate=learn_rate,
                                                          loss_name=loss_name,
                                                          decay_steps=train_iters, decay_rate=0.95)
        init = tf.global_variables_initializer()
        log_txt = open('log.txt', 'w')
        best_acc = 0
        ep = 0
        with tf.Session() as sess:
            sess.run(init)
            if restore:
                self.sess_restore(sess)
                print('restore model')
            for epoch in range(epochs):
                if self.data_provider.shuffle_data:
                    self.data_provider.shuffle()
                for iter in range(train_iters):
                    batch_x, batch_y = self.data_provider.next_batch()
                    _, loss, accuracy = sess.run((self.optimizer, self.loss, self.accuracy),
                                                 feed_dict={self.x: batch_x,
                                                            self.y: imageUnits.cut_image(batch_y,
                                                                                         [None,
                                                                                          self.output_size,
                                                                                          self.output_size,
                                                                                          None]),
                                                            self.keep_prob: keep_prob})

                    print('epoch ' + str(epoch) + ',iter' + str(iter) + ': loss:' + str(
                        loss) + ', accuracy:' + str(accuracy))
                    # print('epoch ' + str(epoch) + ',iter' + str(iter) + ': loss:' + str(
                    # loss) + ', accuracy:' + str(accuracy), file=log_txt)
                    if (iter + 1) % 10 == 0:
                        self.test(sess, name='epoch' + str(epoch) + 'iter' + str(iter))

                    if (epoch * train_iters + iter + 1) % save_steps == 0:
                        acc = self.verify(sess)
                        print('Verify accuracy:' + str(acc))
                        if acc > best_acc:
                            self.save_sess(sess, 'model_epoch' + str(epoch) + '_iter' + str(iter) + '.ckpt')
                            best_acc = acc
                            ep = epoch
                            print('save model, accuracy:' + str(acc))
            self.save_sess(sess, 'last_model.ckpt')
            log_txt.close()
        return best_acc, ep

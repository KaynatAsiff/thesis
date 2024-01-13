#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import inference, GaussianPyramid

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

num_pyramids = 5
learning_rate = 1e-3
iterations = 100000
batch_size = 10
num_channels = 3
patch_size = 80
save_model_path = './model/'
model_name = 'model-epoch'

input_path = './TrainData/input/'
gt_path = './TrainData/label/'

def _parse_function(input_path, gt_path, patch_size=patch_size):
    image_string = tf.io.read_file(input_path)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    rainy = tf.cast(image_decoded, tf.float32) / 255.0

    image_string = tf.io.read_file(gt_path)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    label = tf.cast(image_decoded, tf.float32) / 255.0

    t = time.time()
    Data = tf.image.random_crop(rainy, [patch_size, patch_size, 3], seed=int(t))
    Label = tf.image.random_crop(label, [patch_size, patch_size, 3], seed=int(t))

    return Data, Label

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()

    input_files = os.listdir(input_path)
    input_files = [os.path.join(input_path, file) for file in input_files]

    label_files = os.listdir(gt_path)
    label_files = [os.path.join(gt_path, file) for file in label_files]

    input_files = tf.convert_to_tensor(input_files, dtype=tf.string)
    label_files = tf.convert_to_tensor(label_files, dtype=tf.string)

    dataset = tf.data.Dataset.from_tensor_slices((input_files, label_files))
    dataset = dataset.map(_parse_function)
    dataset = dataset.prefetch(buffer_size=batch_size * 10)
    dataset = dataset.batch(batch_size).repeat()

    # Use `tf.compat.v1.data.make_one_shot_iterator` for non-eager execution
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    inputs, labels = iterator.get_next()

    k = np.float32([.0625, .25, .375, .25, .0625])
    k = np.outer(k, k)
    kernel = k[:, :, None, None] / k.sum() * np.eye(3, dtype=np.float32)
    labels_GaussianPyramid = GaussianPyramid(labels, kernel, (num_pyramids - 1))

    outout_pyramid = inference(inputs)

    loss1 = tf.reduce_mean(tf.abs(outout_pyramid[0] - labels_GaussianPyramid[0]))
    loss2 = tf.reduce_mean(tf.abs(outout_pyramid[1] - labels_GaussianPyramid[1]))
    loss3 = tf.reduce_mean(tf.abs(outout_pyramid[2] - labels_GaussianPyramid[2]))

    loss41 = tf.reduce_mean(tf.abs(outout_pyramid[3] - labels_GaussianPyramid[3]))
    loss42 = tf.reduce_mean((1. - tf.image.ssim(outout_pyramid[3], labels_GaussianPyramid[3], max_val=1.0)) / 2.)

    loss51 = tf.reduce_mean(tf.abs(outout_pyramid[4] - labels))
    loss52 = tf.reduce_mean((1. - tf.image.ssim(outout_pyramid[4], labels, max_val=1.0)) / 2.)

    loss = loss1 + loss2 + loss3 + loss41 + loss42 + loss51 + loss52

    g_optimizer = tf.keras.optimizers.Adam(learning_rate)
    trainable_variables = tf.compat.v1.trainable_variables()

    with tf.compat.v1.name_scope('compute_gradients'):
        grads = tf.gradients(loss, trainable_variables)
        grads_and_vars = list(zip(grads, trainable_variables))

    g_optim = g_optimizer.apply_gradients(grads_and_vars)

    all_vars = tf.compat.v1.trainable_variables()
    saver = tf.compat.v1.train.Saver(var_list=all_vars, max_to_keep=5)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.group(tf.compat.v1.global_variables_initializer(),
                          tf.compat.v1.local_variables_initializer()))
        tf.compat.v1.get_default_graph().finalize()

        if tf.compat.v1.train.get_checkpoint_state(save_model_path):
            ckpt = tf.compat.v1.train.latest_checkpoint(save_model_path)
            saver.restore(sess, ckpt)
            ckpt_num = re.findall(r'(\w*[0-9]+)\w*', ckpt)
            start_point = int(ckpt_num[0]) + 1
            print("loaded successfully")
        else:
            print("re-training")
            start_point = 0

        check_input, check_label = sess.run([inputs, labels])
        print("check patch pair:")
        plt.subplot(1, 3, 1)
        plt.imshow(check_input[0, :, :, :])
        plt.title('input')
        plt.subplot(1, 3, 2)
        plt.imshow(check_label[0, :, :, :])
        plt.title('ground truth')
        plt.show()

        start = time.time()

        for j in range(start_point, iterations):
            _, Training_Loss = sess.run([g_optim, loss])

            if np.mod(j + 1, 100) == 0 and j != 0:
                end = time.time()
                print('%d / %d iterations, Training Loss  = %.4f, running time = %.1f s'
                      % (j + 1, iterations, Training_Loss, (end - start)))
                save_path_full = os.path.join(save_model_path, model_name)
                saver.save(sess, save_path_full, global_step=j + 1)
                start = time.time()

        print('Training finished')
    sess.close()

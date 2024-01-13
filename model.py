# #!/usr/bin/env python2
# # -*- coding: utf-8 -*-
#
# # This is a implementation of our LPNet structure:
# # X. Fu, B. Liang, Y. Huang, X. Ding and J. Paisley. “Lightweight Pyramid Networks for Image Deraining”, IEEE Transactions on Neural Networks and Learning Systems, 2019.
# # author: Xueyang Fu (xyfu@ustc.edu.cn)
#
# import numpy as np
# import tensorflow as tf
#
# num_pyramids = 5  # number of pyramid levels
# num_blocks = 5  # number of recursive blocks
# num_feature = 16  # number of feature maps
# num_channels = 3  # number of input's channels
#
#
# # leaky ReLU
# def lrelu(x, leak=0.2, name='lrelu'):
#     return tf.maximum(x, leak * x, name=name)
#
#
# # Laplacian and Gaussian Pyramid
# def lap_split(img, kernel):
#     with tf.name_scope('split'):
#         low = tf.nn.conv2d(img, kernel, [1, 2, 2, 1], 'SAME')
#         low_upsample = tf.nn.conv2d_transpose(low, kernel * 4, tf.shape(img), [1, 2, 2, 1])
#         high = img - low_upsample
#     return low, high
#
#
# def LaplacianPyramid(img, kernel, n):
#     levels = []
#     for i in range(n):
#         img, high = lap_split(img, kernel)
#         levels.append(high)
#     levels.append(img)
#     return levels[::-1]
#
#
# def GaussianPyramid(img, kernel, n):
#     levels = []
#     low = img
#     for i in range(n):
#         low = tf.nn.conv2d(low, kernel, [1, 2, 2, 1], 'SAME')
#         levels.append(low)
#     return levels[::-1]
#
#
# # create kernel
# def create_kernel(name, shape, initializer=tf.keras.initializers.GlorotNormal()):
#     regularizer = tf.keras.regularizers.l2(l=1e-4)
#     new_variables = tf.Variable(initializer(shape), name=name, regularizer=regularizer)
#     return new_variables
#
#
# # sub network
# def subnet(images, num_feature):
#     kernel0 = create_kernel(name='weights_0', shape=[3, 3, num_channels, num_feature])
#     biases0 = tf.Variable(tf.constant(0.0, shape=[num_feature], dtype=tf.float32), trainable=True, name='biases_0')
#
#     kernel1 = create_kernel(name='weights_1', shape=[3, 3, num_feature, num_feature])
#     biases1 = tf.Variable(tf.constant(0.0, shape=[num_feature], dtype=tf.float32), trainable=True, name='biases_1')
#
#     kernel2 = create_kernel(name='weights_2', shape=[1, 1, num_feature, num_feature])
#     biases2 = tf.Variable(tf.constant(0.0, shape=[num_feature], dtype=tf.float32), trainable=True, name='biases_2')
#
#     kernel3 = create_kernel(name='weights_3', shape=[3, 3, num_feature, num_feature])
#     biases3 = tf.Variable(tf.constant(0.0, shape=[num_feature], dtype=tf.float32), trainable=True, name='biases_3')
#
#     kernel4 = create_kernel(name='weights_4', shape=[1, 1, num_feature, num_channels])
#     biases4 = tf.Variable(tf.constant(0.0, shape=[num_channels], dtype=tf.float32), trainable=True, name='biases_4')
#
#     #  1st layer
#     with tf.name_scope('1st_layer'):
#         conv0 = tf.nn.conv2d(images, kernel0, [1, 1, 1, 1], padding='SAME')
#         bias0 = tf.nn.bias_add(conv0, biases0)
#         bias0 = lrelu(bias0)  # leaky ReLU
#
#         out_block = bias0
#
#     #  recursive blocks
#     for i in range(num_blocks):
#         with tf.name_scope('block_%s' % (i + 1)):
#             conv1 = tf.nn.conv2d(out_block, kernel1, [1, 1, 1, 1], padding='SAME')
#             bias1 = tf.nn.bias_add(conv1, biases1)
#             bias1 = lrelu(bias1)
#
#             conv2 = tf.nn.conv2d(bias1, kernel2, [1, 1, 1, 1], padding='SAME')
#             bias2 = tf.nn.bias_add(conv2, biases2)
#             bias2 = lrelu(bias2)
#
#             conv3 = tf.nn.conv2d(bias2, kernel3, [1, 1, 1, 1], padding='SAME')
#             bias3 = tf.nn.bias_add(conv3, biases3)
#             bias3 = lrelu(bias3)
#
#             out_block = tf.add(bias3, bias0)  # shortcut
#
#     #  reconstruction layer
#     with tf.name_scope('recons'):
#         conv = tf.nn.conv2d(out_block, kernel4, [1, 1, 1, 1], padding='SAME')
#         recons = tf.nn.bias_add(conv, biases4)
#
#         final_out = tf.add(recons, images)  # shortcut
#
#     return final_out
#
#
# # LPNet structure
# def inference(images):
#     with tf.name_scope('inference'):
#         k = np.float32([.0625, .25, .375, .25, .0625])  # Gaussian kernel for image pyramid
#         k = np.outer(k, k)
#         kernel = k[:, :, None, None] / k.sum() * np.eye(3, dtype=np.float32)
#         pyramid = LaplacianPyramid(images, kernel, (num_pyramids - 1))  # rainy Laplacian pyramid
#
#         # subnet 1
#         with tf.name_scope('subnet1'):
#             out1 = subnet(pyramid[0], int((num_feature) / 16))
#             out1 = tf.nn.relu(out1)
#             out1_t = tf.nn.conv2d_transpose(out1, kernel * 4, tf.shape(pyramid[1]), [1, 2, 2, 1])
#
#         # subnet 2
#         with tf.name_scope('subnet2'):
#             out2 = subnet(pyramid[1], int((num_feature) / 8))
#             out2 = tf.add(out2, out1_t)
#             out2 = tf.nn.relu(out2)
#             out2_t = tf.nn.conv2d_transpose(out2, kernel * 4, tf.shape(pyramid[2]), [1, 2, 2, 1])
#
#         # subnet 3
#         with tf.name_scope('subnet3'):
#             out3 = subnet(pyramid[2], int((num_feature) / 4))
#             out3 = tf.add(out3, out2_t)
#             out3 = tf.nn.relu(out3)
#             out3_t = tf.nn.conv2d_transpose(out3, kernel * 4, tf.shape(pyramid[3]), [1, 2, 2, 1])
#
#         # subnet 4
#         with tf.name_scope('subnet4'):
#             out4 = subnet(pyramid[3], int((num_feature) / 2))
#             out4 = tf.add(out4, out3_t)
#             out4 = tf.nn.relu(out4)
#             out4_t = tf.nn.conv2d_transpose(out4, kernel * 4, tf.shape(pyramid[4]), [1, 2, 2, 1])
#
#         # subnet 5
#         with tf.name_scope('subnet5'):
#             out5 = subnet(pyramid[4], int(num_feature))
#             out5 = tf.add(out5, out4_t)
#             out5 = tf.nn.relu(out5)
#
#         outout_pyramid = []
#         outout_pyramid.append(out1)
#         outout_pyramid.append(out2)
#         outout_pyramid.append(out3)
#         outout_pyramid.append(out4)
#         outout_pyramid.append(out5)
#
#         return outout_pyramid
#
#
# if __name__ == '__main__':
#     tf.compat.v1.disable_eager_execution()  # Disable eager execution for compatibility
#
#     input_x = tf.placeholder(tf.float32, [10, None, None, 3])
#
#     outout_pyramid = inference(input_x)
#     var_list = tf.compat.v1.trainable_variables()
#     print("Total parameters' number: %d" % (np.sum([np.prod(v.get_shape().as_list()) for v in var_list])))

#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

num_pyramids = 5
num_blocks = 5
num_feature = 16
num_channels = 3

# leaky ReLU
def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(x, leak * x, name=name)

# Laplacian and Gaussian Pyramid
def lap_split(img, kernel):
    with tf.name_scope('split'):
        low = tf.nn.conv2d(img, kernel, [1, 2, 2, 1], 'SAME')
        low_upsample = tf.nn.conv2d_transpose(low, kernel * 4, tf.shape(img), [1, 2, 2, 1])
        high = img - low_upsample
    return low, high

def LaplacianPyramid(img, kernel, n):
    levels = []
    for i in range(n):
        img, high = lap_split(img, kernel)
        levels.append(high)
    levels.append(img)
    return levels[::-1]

def GaussianPyramid(img, kernel, n):
    levels = []
    low = img
    for i in range(n):
        low = tf.nn.conv2d(low, kernel, [1, 2, 2, 1], 'SAME')
        levels.append(low)
    return levels[::-1]

# create kernel
def create_kernel(name, shape, initializer=tf.keras.initializers.GlorotNormal()):
    regularizer = tf.keras.regularizers.l2(l=1e-4)
    new_variables = tf.Variable(initializer(shape), name=name, regularizer=regularizer)
    return new_variables

# sub network
def subnet(images, num_feature):
    kernel0 = create_kernel(name='weights_0', shape=[3, 3, num_channels, num_feature])
    biases0 = tf.Variable(tf.constant(0.0, shape=[num_feature], dtype=tf.float32), trainable=True, name='biases_0')

    kernel1 = create_kernel(name='weights_1', shape=[3, 3, num_feature, num_feature])
    biases1 = tf.Variable(tf.constant(0.0, shape=[num_feature], dtype=tf.float32), trainable=True, name='biases_1')

    kernel2 = create_kernel(name='weights_2', shape=[1, 1, num_feature, num_feature])
    biases2 = tf.Variable(tf.constant(0.0, shape=[num_feature], dtype=tf.float32), trainable=True, name='biases_2')

    kernel3 = create_kernel(name='weights_3', shape=[3, 3, num_feature, num_feature])
    biases3 = tf.Variable(tf.constant(0.0, shape=[num_feature], dtype=tf.float32), trainable=True, name='biases_3')

    kernel4 = create_kernel(name='weights_4', shape=[1, 1, num_feature, num_channels])
    biases4 = tf.Variable(tf.constant(0.0, shape=[num_channels], dtype=tf.float32), trainable=True, name='biases_4')

    #  1st layer
    with tf.name_scope('1st_layer'):
        conv0 = tf.nn.conv2d(images, kernel0, [1, 1, 1, 1], padding='SAME')
        bias0 = tf.nn.bias_add(conv0, biases0)
        bias0 = lrelu(bias0)  # leaky ReLU

        out_block = bias0

    #  recursive blocks
    for i in range(num_blocks):
        with tf.name_scope('block_%s' % (i + 1)):
            conv1 = tf.nn.conv2d(out_block, kernel1, [1, 1, 1, 1], padding='SAME')
            bias1 = tf.nn.bias_add(conv1, biases1)
            bias1 = lrelu(bias1)

            conv2 = tf.nn.conv2d(bias1, kernel2, [1, 1, 1, 1], padding='SAME')
            bias2 = tf.nn.bias_add(conv2, biases2)
            bias2 = lrelu(bias2)

            conv3 = tf.nn.conv2d(bias2, kernel3, [1, 1, 1, 1], padding='SAME')
            bias3 = tf.nn.bias_add(conv3, biases3)
            bias3 = lrelu(bias3)

            out_block = tf.add(bias3, bias0)  # shortcut

    #  reconstruction layer
    with tf.name_scope('recons'):
        conv = tf.nn.conv2d(out_block, kernel4, [1, 1, 1, 1], padding='SAME')
        recons = tf.nn.bias_add(conv, biases4)

        final_out = tf.add(recons, images)  # shortcut

    return final_out

# GAN Architecture
def generator(images, num_feature):
    with tf.name_scope('generator'):
        kernel = create_kernel(name='gen_weights', shape=[3, 3, num_channels, num_feature])
        biases = tf.Variable(tf.constant(0.0, shape=[num_channels], dtype=tf.float32), trainable=True, name='gen_biases')

        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        generated_images = tf.nn.bias_add(conv, biases)

        return generated_images

def discriminator(images):
    with tf.name_scope('discriminator'):
        kernel = create_kernel(name='dis_weights', shape=[3, 3, num_channels, num_feature])
        biases = tf.Variable(tf.constant(0.0, shape=[num_feature], dtype=tf.float32), trainable=True, name='dis_biases')

        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        validity = tf.nn.bias_add(conv, biases)

        return validity

# GAN Training
def gan_training(real_images, lpnet_output):
    generator_model = generator(lpnet_output, num_feature)
    discriminator_model_real = discriminator(real_images)
    discriminator_model_fake = discriminator(generator_model)

    # Define GAN loss and training operations
    gan_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=discriminator_model_real, labels=tf.ones_like(discriminator_model_real)))
    gan_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=discriminator_model_fake, labels=tf.zeros_like(discriminator_model_fake)))
    gan_loss = gan_loss_real + gan_loss_fake

    generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=discriminator_model_fake, labels=tf.ones_like(discriminator_model_fake)))

    # Define GAN training operations
    gan_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator') + \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
    gan_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)
    gan_train_op = gan_optimizer.minimize(gan_loss, var_list=gan_vars)

    return gan_loss, generator_loss, gan_train_op, generator_model

# Main LPNet structure
def inference(images):
    with tf.name_scope('inference'):
        k = np.float32([.0625, .25, .375, .25, .0625])
        k = np.outer(k, k)
        kernel = k[:, :, None, None] / k.sum() * np.eye(3, dtype=np.float32)
        pyramid = LaplacianPyramid(images, kernel, (num_pyramids - 1))

        with tf.name_scope('gan_training'):
            gan_loss, generator_loss, gan_train_op, generator_model = gan_training(images, pyramid[-1])

        return pyramid, gan_loss, generator_loss, gan_train_op, generator_model

if __name__ == '__main__':
    tf.compat.v1.disable_eager_execution()

    input_x = tf.placeholder(tf.float32, [10, None, None, 3])

    outout_pyramid, gan_loss, generator_loss, gan_train_op, generator_model = inference(input_x)
    var_list = tf.compat.v1.trainable_variables()
    print("Total parameters' number: %d" % (np.sum([np.prod(v.get_shape().as_list()) for v in var_list])))

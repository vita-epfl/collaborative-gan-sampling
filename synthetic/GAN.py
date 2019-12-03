from __future__ import division
import tensorflow as tf

class GAN(object):

    def __init__(self, sess, args, data, noise):
        self.sess = sess
        self.nhidden = args.nhidden
        self.nlayers = args.nlayers
        self.niters = args.niters
        self.batch_size = args.batch_size
        self.z_dim = 2
        self.lrg = args.lrg
        self.lrd = args.lrd
        self.data = data
        self.noise = noise
        self.scale = args.scale

    def minibatch(self, x, num_kernels=5, kernel_dim=3):
        net = tf.layers.dense(inputs=x, units=num_kernels*kernel_dim, name='minibatch')
        activation = tf.reshape(net, (-1, num_kernels, kernel_dim))
        diffs = tf.expand_dims(activation, 3) - \
            tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
        return tf.concat([x, minibatch_features], 1)

    def discriminator(self, x, is_training=True, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            net = tf.layers.dense(inputs=x, units=self.nhidden, activation=None, name='d_fc1')
            net = tf.nn.relu(net, name='d_rl1')
            for i in range(self.nlayers-2):
                net = tf.layers.dense(inputs=net, units=self.nhidden, activation=None, name='d_fc'+str(i+2))
                net = tf.nn.relu(net, name='d_rl'+str(i+2))
            out_logit = tf.layers.dense(inputs=net, units=1, name='d_fc'+str(self.nlayers))
            out = tf.nn.sigmoid(out_logit)
            return out, out_logit

    def generator(self, z, is_training=True, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            net = tf.layers.dense(inputs=z, units=self.nhidden, activation=None, name='g_fc1')
            net = tf.contrib.layers.batch_norm(net, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training)
            net = tf.nn.relu(net, name='g_rl1')
            for i in range(self.nlayers-2):
                net = tf.layers.dense(inputs=net, units=self.nhidden, activation=None, name='g_fc'+str(i+2))
                net = tf.contrib.layers.batch_norm(net, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=is_training)
                net = tf.nn.relu(net, name='g_rl'+str(i+2))
            out = tf.layers.dense(inputs=net, units=2, activation=None, name='g_fc'+str(self.nlayers))
            return out

    def build_model(self):
        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [None, 2], name='real_placeholder')

        # noises
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z_placeholder')

        """ Loss Function """

        # output of D for real images
        d_real, d_real_logits = self.discriminator(self.inputs, is_training=True, reuse=False)

        # output of D for fake images
        self.generates = self.generator(self.z, is_training=True, reuse=False)
        d_fake, d_fake_logits = self.discriminator(self.generates, is_training=True, reuse=True)

        # get loss for discriminator
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits, labels=tf.ones_like(d_real)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.zeros_like(d_fake)))

        self.d_loss = self.d_loss_real + self.d_loss_fake

        # get loss for generator
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.ones_like(d_fake)))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        ''' Plugin '''
        self.grad_default = tf.gradients(self.g_loss, self.generates)[0]
        # grads_collected = tf.gradients(self.generates, g_vars, self.grad_default)

        self.grad_plugin = tf.placeholder(tf.float32, [None, 2])
        grads_collected = tf.gradients(self.generates, g_vars, self.grad_plugin)
        grads_and_vars_collected = list(zip(grads_collected, g_vars))

        self.grad_z = tf.gradients(self.g_loss, self.z)[0]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.GradientDescentOptimizer(self.lrd) \
                      .minimize(self.d_loss, var_list=d_vars)

            self.g_optim = tf.train.GradientDescentOptimizer(self.lrg).apply_gradients(grads_and_vars_collected)

        """" Testing """
        # for test
        self.fake_samples = self.generator(self.z, is_training=False, reuse=True)

        # self.batch_samples = tf.placeholder(tf.float32, [None, 2], name='batch_placeholder')
        self.fake_sigmoid, self.fake_logit = self.discriminator(self.fake_samples, is_training=False, reuse=True)
        self.fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit, labels=tf.ones_like(self.fake_sigmoid)))
        self.fake_saliency = tf.gradients(self.fake_loss, self.fake_samples)[0]

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

        # saver to save model
        self.saver = tf.train.Saver(max_to_keep=20)

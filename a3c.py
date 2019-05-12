import gym
import os
import tensorflow as tf
from scipy.misc import imresize
from tensorboardX import SummaryWriter

class A3CNetwork(object):

    def __init__(self, name, input_shape, output_dim, logdir=None):
        """Network structure is defined here
        Args:
            name (str): The name of scope
            input_shape (list): The shape of input image [H, W, C]
            output_dim (int): Number of actions
            logdir (str, optional): directory to save summaries
                TODO: create a summary op
        """
        with tf.variable_scope(name):
            self.states = tf.placeholder(tf.float32, shape=[None, *input_shape], name="states")
            self.actions = tf.placeholder(tf.uint8, shape=[None], name="actions")
            self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
            self.advantage = tf.placeholder(tf.float32, shape=[None], name="advantage")

            action_onehot = tf.one_hot(self.actions, output_dim, name="action_onehot")
            net = self.states

            net = tf.layers.conv2d(inputs=net, filters=16, kernel_size=[8, 8], strides=[4, 4], padding='VALID', activation=tf.nn.relu)
            net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=[4, 4], strides=[2, 2], padding='VALID', activation=tf.nn.relu)
            
            net = tf.layers.flatten(net)
            net = tf.layers.dense(inputs=net, units=256, activation=tf.nn.relu)
            
            self.action_prob = tf.layers.dense(inputs=net, units=output_dim, activation=tf.nn.softmax)
            self.values = tf.squeeze(tf.layers.dense(inputs=net, units=1, activation=None))

            single_action_prob = tf.reduce_sum(self.action_prob * action_onehot, axis=1)
            clip_single_action_prob = tf.clip_by_value(single_action_prob, 1e-7, 1.0)
            entropy = - self.action_prob * tf.log(self.action_prob + 1e-7)
            entropy = tf.reduce_sum(entropy, axis=1)

            log_action_prob = tf.log(clip_single_action_prob)
            maximize_objective = log_action_prob * self.advantage + entropy * 0.01
            self.actor_loss = -tf.reduce_mean(maximize_objective)

            self.value_loss = tf.losses.mean_squared_error(labels=self.rewards, predictions=self.values)

            # tensorboardX
            self.entropy = tf.reduce_mean(entropy)
            self.pi_loss = tf.reduce_mean(log_action_prob * self.advantage)
            self.mean_value_loss = tf.reduce_mean(self.value_loss)

            # optimization
            self.total_loss = self.actor_loss + self.value_loss * .5
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            #self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.gradients = self.optimizer.compute_gradients(self.total_loss, var_list)
        self.gradients_placeholders = []

        for grad, var in self.gradients:
            self.gradients_placeholders.append((tf.placeholder(var.dtype, shape=var.get_shape()), var))
        self.apply_gradients = self.optimizer.apply_gradients(self.gradients_placeholders)
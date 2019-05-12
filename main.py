import tensorflow as tf
import numpy as np
import threading
import gym
import os
import multiprocessing
from scipy.misc import imresize
from tensorboardX import SummaryWriter
import a3c
import async_agent


def main():
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    coord = tf.train.Coordinator()

    n_threads = multiprocessing.cpu_count()
    input_shape = [80, 80, 4]
    output_dim = 3  # {1, 2, 3}
    global_network = a3c.A3CNetwork(name="global",
                                input_shape=input_shape,
                                output_dim=output_dim)

    thread_list = []
    env_list = []

    for id in range(n_threads):
        env = gym.make("Pong-v0")

        single_agent = async_agent.Agent(env=env,
                                session=sess,
                                coord=coord,
                                name="thread_{}".format(id),
                                global_network=global_network,
                                input_shape=input_shape,
                                output_dim=output_dim)
        thread_list.append(single_agent)
        env_list.append(env)

    init = tf.global_variables_initializer()
    sess.run(init)

    for t in thread_list:
        t.start()


if __name__ == '__main__':
    main()
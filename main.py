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
    try:
        tf.reset_default_graph()
        sess = tf.InteractiveSession()
        coord = tf.train.Coordinator()

        checkpoint_dir = "checkpoint"
        monitor_dir = "monitors"
        save_path = os.path.join(checkpoint_dir, "model.ckpt")

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            print("Directory {} was created".format(checkpoint_dir))

        n_threads = multiprocessing.cpu_count()
        input_shape = [80, 80, 1]
        output_dim = 3  # {1, 2, 3}
        global_network = a3c.A3CNetwork(name="global",
                                    input_shape=input_shape,
                                    output_dim=output_dim)

        thread_list = []
        env_list = []

        for id in range(n_threads):
            env = gym.make("Pong-v0")

            if id == 0:
                env = gym.wrappers.Monitor(env, monitor_dir, force=True)

            single_agent = async_agent.Agent(env=env,
                                 session=sess,
                                 coord=coord,
                                 name="thread_{}".format(id),
                                 global_network=global_network,
                                 input_shape=input_shape,
                                 output_dim=output_dim)
            thread_list.append(single_agent)
            env_list.append(env)

        if tf.train.get_checkpoint_state(os.path.dirname(save_path)):
            var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
            saver = tf.train.Saver(var_list=var_list)
            saver.restore(sess, save_path)
            print("Model restored to global")
        else:
            init = tf.global_variables_initializer()
            sess.run(init)
            print("No model is found")

        for t in thread_list:
            t.start()

        print("Ctrl + C to close")
        coord.wait_for_stop()

    except KeyboardInterrupt:
        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "global")
        saver = tf.train.Saver(var_list=var_list)
        saver.save(sess, save_path)
        print('Checkpoint Saved to {}'.format(save_path))

        print("Closing threads")
        coord.request_stop()
        coord.join(thread_list)

        print("Closing environments")
        for env in env_list:
            env.close()

        sess.close()


if __name__ == '__main__':
    main()
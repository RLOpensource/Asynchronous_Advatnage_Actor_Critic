import gym
import os
from scipy.misc import imresize
from tensorboardX import SummaryWriter
import threading
import utils
import a3c
import tensorflow as tf
import numpy as np
import copy

class Agent(threading.Thread):

    def __init__(self, session, env, coord, name, global_network, input_shape, output_dim, logdir=None):
        """Agent worker thread
        Args:
            session (tf.Session): Tensorflow session needs to be shared
            env (gym.env): Gym environment
            coord (tf.train.Coordinator): Tensorflow Queue Coordinator
            name (str): Name of this worker
            global_network (A3CNetwork): Global network that needs to be updated
            input_shape (list): Required for local A3CNetwork (H, W, C)
            output_dim (int): Number of actions
            logdir (str, optional): If logdir is given, will write summary
                TODO: Add summary
        """
        super(Agent, self).__init__()
        self.local = a3c.A3CNetwork(name, input_shape, output_dim, logdir)
        self.global_to_local = utils.copy_src_to_dst("global", name)
        self.global_network = global_network

        self.input_shape = input_shape
        self.output_dim = output_dim
        self.env = env
        self.sess = session
        self.coord = coord
        self.name = name
        self.logdir = logdir
        self.episode = 0
        self.writer = SummaryWriter('runs/'+self.name)

    def print(self, reward, max_prob, pi_loss, value_loss, entropy, episode_step):
        self.writer.add_scalar('score', reward, self.episode)
        self.writer.add_scalar('max_prob', max_prob, self.episode)
        self.writer.add_scalar('pi_loss', pi_loss, self.episode)
        self.writer.add_scalar('value_loss', value_loss, self.episode)
        self.writer.add_scalar('entropy', entropy, self.episode)
        self.writer.add_scalar('episode_step', episode_step, self.episode)
        message = "Agent(name={}, episode={}, reward={}, max_prob={})".format(self.name, self.episode, reward, max_prob)
        print(message)

    def play_episode(self):
        self.sess.run(self.global_to_local)

        states = []
        actions = []
        rewards = []
        dones = []


        s = self.env.reset()
        s = utils.pipeline(s)
        history = np.stack((s, s, s, s), axis=2)

        done = False
        total_reward = 0
        time_step = 0
        self.episode += 1
        episode_step = 0
        total_max_prob = 0
        total_pi_loss = 0
        total_entropy = 0
        total_value_loss = 0

        while not done:

            a, max_prob = self.choose_action(copy.deepcopy(history))

            total_max_prob += max_prob
            episode_step += 1

            s2, r, done, _ = self.env.step(int(a+1))

            s2 = utils.pipeline(s2)
            total_reward += r

            states.append(copy.deepcopy(history))
            actions.append(a)
            rewards.append(r)

            history[:, :, :3] = history[:, :, 1:]
            history[:, :, 3] = s2

            if r == -1 or r == 1 or done:
                time_step += 1

                if time_step >= 5 or done:
                    pi_loss, value_loss, entropy = self.train(states, actions, rewards, done)
                    total_pi_loss += pi_loss
                    total_value_loss += value_loss
                    total_entropy += entropy
                    self.sess.run(self.global_to_local)
                    states, actions, rewards = [], [], []
                    time_step = 0

        total_max_prob /= episode_step
        self.print(total_reward, total_max_prob, total_pi_loss, total_value_loss, total_entropy, episode_step)

    def run(self):
        self.sess.run(self.global_to_local)
        
        s = self.env.reset()
        s = utils.pipeline(s)
        history = np.stack((s, s, s, s), axis=2)

        done = False
        total_reward = 0
        time_step = 0
        self.episode = 0
        episode_step = 0
        total_max_prob = 0
        total_pi_loss = 0
        total_entropy = 0
        total_value_loss = 0
        train_step = 0

        while True:
            train_step += 1
            states = []
            actions = []
            rewards = []
            dones = []
            for i in range(256):

                a, max_prob = self.choose_action(copy.deepcopy(history))
                total_max_prob += max_prob
                episode_step += 1

                s2, r, real_done, _ = self.env.step(int(a+1))

                s2 = utils.pipeline(s2)
                total_reward += r

                d = False
                if r == -1 or r == 1:
                    d = True

                states.append(copy.deepcopy(history))
                actions.append(a)
                rewards.append(r)
                dones.append(d)

                history[:, :, :3] = history[:, :, 1:]
                history[:, :, 3] = s2

                if real_done:
                    self.writer.add_scalar('score', total_reward, self.episode)
                    self.writer.add_scalar('max_prob', total_max_prob / episode_step, self.episode)
                    self.writer.add_scalar('episode_step', episode_step, self.episode)
                    
                    print(self.name, total_reward, total_max_prob / episode_step, episode_step)
                    s = self.env.reset()
                    s = utils.pipeline(s)
                    history = np.stack((s, s, s, s), axis=2)

                    done = False
                    total_reward = 0
                    time_step = 0
                    self.episode += 1
                    episode_step = 0
                    total_max_prob = 0
                    total_pi_loss = 0
                    total_entropy = 0
                    total_value_loss = 0

            pi_loss, value_loss, entropy = self.train_with_done(states, actions, rewards, dones)
            self.sess.run(self.global_to_local)
            self.writer.add_scalar('pi_loss', pi_loss, train_step)
            self.writer.add_scalar('value_loss', value_loss, train_step)
            self.writer.add_scalar('entropy', entropy, train_step)

    def choose_action(self, states):
        """
        Args:
            states (2-D array): (N, H, W, 1)
        """
        states = np.reshape(states, [-1, *self.input_shape])
        feed = {
            self.local.states: states
        }

        action = self.sess.run(self.local.action_prob, feed)
        action = np.squeeze(action)

        act = np.random.choice(self.output_dim, p=action)

        return act, max(action)

    def train(self, states, actions, rewards, done):
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        feed = {
            self.local.states: states
        }

        values = self.sess.run(self.local.values, feed)

        rewards = utils.discount_reward(rewards, gamma=0.99)

        advantage = rewards - values
        advantage -= np.mean(advantage)
        advantage /= np.std(advantage) + 1e-8

        feed = {
            self.local.states: states,
            self.local.actions: actions,
            self.local.rewards: rewards,
            self.local.advantage: advantage
        }

        gradients, pi_loss, value_loss, entropy = self.sess.run([self.local.gradients, self.local.pi_loss, self.local.mean_value_loss, self.local.entropy], feed)

        feed = []
        for (grad, _), (placeholder, _) in zip(gradients, self.global_network.gradients_placeholders):
            feed.append((placeholder, grad))

        feed = dict(feed)
        self.sess.run(self.global_network.apply_gradients, feed)

        return pi_loss, value_loss, entropy

    def train_with_done(self, states, actions, rewards, dones):
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        feed = {
            self.local.states: states
        }

        values = self.sess.run(self.local.values, feed)
        rewards = utils.discount_reward_with_done(rewards, dones, gamma=0.99)
        
        advantage = rewards - values
        advantage -= np.mean(advantage)
        advantage /= np.std(advantage) + 1e-8

        feed = {
            self.local.states: states,
            self.local.actions: actions,
            self.local.rewards: rewards,
            self.local.advantage: advantage
        }

        gradients, pi_loss, value_loss, entropy = self.sess.run([self.local.gradients, self.local.pi_loss, self.local.mean_value_loss, self.local.entropy], feed)
        feed = []
        for (grad, _), (placeholder, _) in zip(gradients, self.global_network.gradients_placeholders):
            feed.append((placeholder, grad))

        feed = dict(feed)
        self.sess.run(self.global_network.apply_gradients, feed)

        return pi_loss, value_loss, entropy

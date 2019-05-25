import tensorflow as tf
import numpy as np
import threading
import gym
import os
from scipy.misc import imresize


def copy_src_to_dst(from_scope, to_scope):
    """Creates a copy variable weights operation
    Args:
        from_scope (str): The name of scope to copy from
            It should be "global"
        to_scope (str): The name of scope to copy to
            It should be "thread-{}"
    Returns:
        list: Each element is a copy operation
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def pipeline(image, new_HW=(80, 80), height_range=(35, 193), bg=(144, 72, 17)):
    """Returns a preprocessed image
    (1) Crop image (top and bottom)
    (2) Remove background & grayscale
    (3) Reszie to smaller image
    Args:
        image (3-D array): (H, W, C)
        new_HW (tuple): New image size (height, width)
        height_range (tuple): Height range (H_begin, H_end) else cropped
        bg (tuple): Background RGB Color (R, G, B)
    Returns:
        image (3-D array): (H, W, 1)
    """
    image = crop_image(image, height_range)
    image = resize_image(image, new_HW)
    image = kill_background_grayscale(image, bg)
    image = np.expand_dims(image, axis=2)
    image = np.reshape(image, new_HW)

    return image


def resize_image(image, new_HW):
    """Returns a resized image
    Args:
        image (3-D array): Numpy array (H, W, C)
        new_HW (tuple): Target size (height, width)
    Returns:
        image (3-D array): Resized image (height, width, C)
    """
    return imresize(image, new_HW, interp="nearest")


def crop_image(image, height_range=(35, 195)):
    """Crops top and bottom
    Args:
        image (3-D array): Numpy image (H, W, C)
        height_range (tuple): Height range between (min_height, max_height)
            will be kept
    Returns:
        image (3-D array): Numpy image (max_H - min_H, W, C)
    """
    h_beg, h_end = height_range
    return image[h_beg:h_end, ...]


def kill_background_grayscale(image, bg):
    """Make the background 0
    Args:
        image (3-D array): Numpy array (H, W, C)
        bg (tuple): RGB code of background (R, G, B)
    Returns:
        image (2-D array): Binarized image of shape (H, W)
            The background is 0 and everything else is 1
    """
    H, W, _ = image.shape

    R = image[..., 0]
    G = image[..., 1]
    B = image[..., 2]

    cond = (R == bg[0]) & (G == bg[1]) & (B == bg[2])

    image = np.zeros((H, W))
    image[~cond] = 1

    return image

def discount_reward_with_done(rewards, done, gamma=0.99):
    """Returns discounted rewards
    Args:
        rewards (1-D array): Reward array
        gamma (float): Discounted rate
    Returns:
        discounted_rewards: same shape as `rewards`
    Notes:
        In Pong, when the reward can be {-1, 0, 1}.
        However, when the reward is either -1 or 1,
        it means the game has been reset.
        Therefore, it's necessaray to reset `running_add` to 0
        whenever the reward is nonzero
    """
    discounted_r = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        if done[t]:
            running_add = 0
        running_add = running_add * gamma + rewards[t]
        discounted_r[t] = running_add

    return discounted_r
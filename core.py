import tensorflow as tf

def cnn_model(net, output_dim, activation, final_activation):
    net = tf.layers.conv2d(inputs=net, filters=16, kernel_size=[8, 8], strides=[4, 4], padding='VALID', activation=activation)
    net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=[4, 4], strides=[2, 2], padding='VALID', activation=activation)
    net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=[3, 3], strides=[1, 1], padding='VALID', activation=activation)

    net = tf.layers.flatten(net)
    net = tf.layers.dense(inputs=net, units=256, activation=activation)

    action_prob = tf.layers.dense(inputs=net, units=output_dim, activation=final_activation)
    values = tf.squeeze(tf.layers.dense(inputs=net, units=1, activation=None))
    return action_prob, values
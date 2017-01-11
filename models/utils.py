import tensorflow as tf

def init_weight(shape, name=None):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)

def init_bias(shape, value=0.1, name=None):
    return tf.Variable(tf.constant(value, shape=[shape]), name=name)

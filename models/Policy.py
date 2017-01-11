from tflearn.layers.core import *
from tflearn.layers.conv import *
import numpy as np
import tensorflow as tf
import tflearn as tl
import utils

FLAGS = tf.app.flags.FLAGS


class PolicyNetworkConv(object):
    def __init__(self, sess=None):
        """
        Using convolutional embeddings and policy gradients to predict good next moves
        """
        self.hidden_size = FLAGS.hidden_size
        self.board_size = FLAGS.image_size
        self.learning_rate = FLAGS.learning_rate
        self.input_size = FLAGS.image_size * FLAGS.image_size + 1
        self.piece_types = 14

        # I/O placeholders
        self.X = tf.placeholder(shape=(None, self.input_size), dtype=tf.float32)
        self.Y = tf.placeholder(shape=(None, 2), dtype=tf.int32)
        self.keep_prob = tf.placeholder(shape=None, dtype=tf.float32)

        # Network itself
        # TODO: Include RL play against prebuilt networks and evaluation from experts
        # eraffig
        self._init_weights()
        self.from_logits, self.to_logits = self._inference_graph()
        self._predict_from, self._predict_to =\
                tf.nn.softmax(self.from_logits), tf.nn.softmax(self.to_logits)
        self.loss = self._loss()
        self.optimizer = self._optimizer()

        # Tensorflow session initialization
        init = tf.initialize_all_variables()
        if sess is not None:
            self.sess = sess
        else:
            self.sess = tf.Session()
        
        self.sess.run(init)
    
    def _init_weights(self):
        self.embeddings = utils.init_weight(shape=[self.piece_types, 1], name='Embeddings')

    def _inference_graph(self):
        with tf.name_scope('policy_network'):
            embedding = tf.nn.embedding_lookup(self.embeddings, self.X, name='embedding')
            reshaped = tf.reshape(embedding[:, :-1], [-1, self.board_size, self.board_size])

            # Two convolutional layers
            conv_1 = conv_2d(reshaped, 32, 3, activation='relu', name='conv_1')
            max_pool_1 = max_pool_2d(conv_1, 2, name='max_pool_1')
            conv_2 = conv_2d(max_pool_1, 64, 3, activation='relu', name='conv_2')
            max_pool_2 = max_pool_2d(conv_2, 2, name='conv_2')
            flattened = flatten(max_pool_2, name='flattening')

            # FC layers
            fc_1 = fully_connected(flattened, self.hidden_size, activation='relu', name='fc_1')
            drop = dropout(fc_1, keep_prob=self.keep_prob)
            fc_2 = fully_connected(fc_1, 128, activation='relu', name='policy')

            return fc_2[:, :64], fc_2[:, 64:]
    
    def _loss(self):
        # TODO: Add policy gradient buffer updates with discounted rewards
        one_hot_y = tf.one_hot(self.y, 64, 1.0, 0.0, axis=2)
        one_hot_y = tf.reshape(one_hot_y, [-1, 64 * 2])

        from_y = one_hot_y[:, :64]
        to_y = one_hot_y[:, 64:]

        loss_from = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(self.from_logits, from_y)
        )
        loss_to = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(self.to_logits, to_y)
        )

        loss = tf.add(loss_from, loss_to, name='loss')
        tf.summary.scalar('Loss', loss)

        return loss

    def _optimizer(self):
        return tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
    
    def partial_fit_step(self, X, Y):
        """
        Train on batch with input X and output Y
        TODO: Add policy gradient changes
        """

        loss, _ = self.sess.run(
            [self.loss, self.optimizer],
            feed_dict={
                self.X: X,
                self.Y: Y,
                self.keep_prob: 0.8
            }
        )

        return loss
    
    def predict(self, X):
        """
        Predict next move from board state X
        """

        predict_from, predict_to = self.sess.run(
            [self._predict_from, self._predict_to],
            feed_dict={
                self.X: X,
                self.keep_prob: 1.0
            }
        )

        return np.argmax(predict_from)[0], np.argmax(predict_to)[0]
    
        
class PolicyNetworkFC(object):
    def __init__(self):
        """
        Fully connected embedding policy network for chess
        """
        self.batch_size = FLAGS.batch_size
        self.input_size = FLAGS.image_size * FLAGS.image_size + 1
        self.hidden_size = FLAGS.hidden_size
        self.learning_rate = FLAGS.learning_rate

    
    def _init_weights(self):
        pass

    def _inference_graph(self):
        pass
        
    def _loss(self):
        pass
    
    def _optimizer(self):
        pass
    
    def partial_fit_step(self, X, Y):
        pass
    
    def predict(self, X):
        pass
    

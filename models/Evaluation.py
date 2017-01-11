from __future__ import division
import tensorflow as tf
import numpy as np
import utils
from tensorflow.contrib import slim


FLAGS = tf.app.flags.FLAGS


class EvaluationNetworkConv(object):
    def __init__(self):
        """
        Evaluation network for chess boards. Based on an MNIST CNN but for 8x8 images
        """
        # Network Parameters
        self.logging = FLAGS.logging
        self.batch_size = FLAGS.batch_size
        self.learning_rate = FLAGS.learning_rate
        self.image_size, self.hidden_size = FLAGS.image_size, FLAGS.hidden_size
        self.image_channels = FLAGS.image_channels
        self.piece_types = 6 + 6 + 1 + 2
        
        # Placeholders for inputs
        self.input_shape = [self.batch_size, self.image_size,\
                            self.image_size, self.image_channels]
        self.x = tf.placeholder(tf.int32, shape=self.input_shape)
        self.y = tf.placeholder(tf.int8, shape=[self.batch_size])
        self.keep_prob = tf.placeholder(tf.float32)

        # Set up weights, network and training logits
        self._init_weights()
        self.logits = self._inference_graph()
        self.loss = self._loss()
        self.optimizer = self._optimizer()

        # Setting up instance
        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)

    def _init_weights(self):
        self.embeddings = utils.init_weight([self.piece_types, 1]) # Create a piecewise embedding
        self.conv1_w = utils.init_weight([3, 3, self.image_channels, 32], name='conv1_weight')
        self.conv1_b = utils.init_bias(32, 0.0, 'conv1_bias')
        
        self.conv2_w = utils.init_weight([3, 3, 32, 64], name='conv2_weight')
        self.conv2_b = utils.init_bias(64, 0.0, name='conv2_bias')

        fc_size = self.image_size // 4 * self.image_size // 4 * 64 * self.image_channels + 1
        self.fcw1 = utils.init_weight([fc_size, self.hidden_size], 'fully_connected_w1')
        self.fcb1 = utils.init_bias(self.hidden_size, 0.1, 'fully_connected_b1')

        self.fcw2 = utils.init_weight([self.hidden_size, 1], name='fully_connected_w2')
        self.fcb2 = utils.init_bias(1, 0.1, 'fully_connected_b2')

    def _inference_graph(self, training):
        embeddings = tf.nn.embedding_lookup(self.embeddings, self.x[:, :-1])
        image = tf.reshape(embeddings, [self.batch_size, 8, 8])
        conv_1 = tf.nn.conv2d(image, self.conv1_w, strides=[1, 1, 1, 1], padding='SAME')
        relu_1 = tf.nn.relu(tf.nn.bias_add(conv_1, self.conv1_b))
        pool_1 = tf.nn.max_pool(relu_1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        conv_2 = tf.nn.conv2d(self.x, self.conv2_w, strides=[1, 1, 1, 1], padding='SAME')
        relu_2 = tf.nn.relu(tf.nn.bias_add(conv_2, self.conv2_b))
        pool_2 = tf.nn.max_pool(relu_2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

        bsz, _, _, _ = pool_2.get_shape().as_list()
        reshaped = tf.reshape(pool_2, [shape[0], -1]) # bsz, flattened
        reshaped_with_turn = tf.concat(2, [self.x[:, -1], reshaped])
        fc1 = tf.matmul(reshaped_with_turn, self.fcw1) + self.fcb1
        relu_fc1 = tf.nn.relu(fc1)

        dropout = tf.nn.dropout(relu_fc1, self.keep_prob)

        fc2 = tf.matmul(dropout, self.fcw2) + self.fcb2
        out = tf.nn.softmax(fc2)

        return out
    
    def _loss(self):
        # TODO: Include L2 regularization later
        # TODO: Add policy gradient buffer updates with discounted rewards
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.y))
    
    def _optimizer(self):
        return tf.train.AdamOptimizer(self.learning_rate)

    def partial_fit_step(self, X, Y):
        """
        Train network on batch input X and Y
        """
        cost, _ = self.sess.run([self.loss, self.optimizer],
                                feed_dict={
                                    self.x: X,
                                    self.y: Y,
                                    self.keep_prob: 0.8
                                })
        
        return cost

    def predict(self, X):
        """
        Predicts either single X or batch, returns probability of white and black winning
        """

        prediction = self.sess.run(self.logits, feed_dict={self.x: X, self.keep_prob: 1.0})

        return prediction
        

class EvaluationNetworkFC(object):
    def __init__(self):
        self.logging = FLAGS.logging
        self.batch_size = FLAGS.batch_size
        self.learning_rate = FLAGS.learning_rate
        self.input_size, self.hidden_size = FLAGS.image_size * FLAGS.image_size + 1, FLAGS.hidden_size
        self.embedding_size = FLAGS.embedding_size
        self.piece_types = 6 + 6 + 1 + 2

        self.x = tf.placeholder(shape=[self.batch_size, self.input_size], dtype=tf.int64)
        self.y = tf.placeholder(shape=[self.batch_size], dtype=tf.int64)

        self._init_weights()
        self.logits = self._inference_graph()
        self._predict_logits = tf.nn.softmax(self.logits, dim=0)
        self.loss = self._loss()
        self.optimizer = self._optimizer()
        
        # Set up training accuracy logging
        if self.logging:
            self._training_accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(self._predict_logits, 1), self.y), "float32")
            )
            tf.scalar_summary('Loss', self.loss)
            tf.scalar_summary('Training Accuracy', self._training_accuracy)
            self.summaries = tf.merge_all_summaries()
        
        # Setting up tensorflow session
        init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(init)
    
    def _init_weights(self):
        self.embeddings = utils.init_weight([self.piece_types, self.embedding_size], name="Embeddings")
        self.fcw1 = utils.init_weight([self.embedding_size * self.input_size, self.hidden_size], name='fc1_weight')
        self.fcb1 = utils.init_bias(self.hidden_size, name='fc1_bias')

        self.fcw2 = utils.init_weight([self.hidden_size, 3], name='fc2_weight')
        self.fcb2 = utils.init_bias(3, name='fc2_bias')
    
    def _inference_graph(self):
        embedding = tf.nn.embedding_lookup(self.embeddings, self.x)
        flattened = tf.reshape(embedding, [self.batch_size, -1])

        fc_layer_1 = tf.matmul(flattened, self.fcw1) + self.fcb1
        relu_1 = tf.nn.relu(fc_layer_1)

        fc_layer_2 = tf.matmul(relu_1, self.fcw2) + self.fcb2
        relu_2 = tf.nn.relu(fc_layer_2)

        return relu_2
    
    def _loss(self):
        one_hot_y = tf.one_hot(self.y, 3, 1.0, 0.0)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, one_hot_y))
    
    def _optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def partial_fit_step(self, X, Y):
        """
        Train network on batch input X, Y. Returns cost.
        """
        cost, _, accuracy, summary = self.sess.run([self.loss, self.optimizer,\
                                                self._training_accuracy, self.summaries],
                                                feed_dict={
                                                    self.x: X,
                                                    self.y: Y
                                                })
            

        return cost, summary, accuracy

    def predict(self, X):
        """
        Predict probability of white/black winning board X, last index is size to move
        """

        prediction = self.sess.run([self._predict_logits],
                                   feed_dict={
                                       self.x: X
                                   })
        
        return prediction

    def accuracy(self, X, Y):
        """
        Gets accuracy of predictions for batch X with true result Y
        """

        loss, accuracy = self.sess.run([self.loss, self._training_accuracy],
                                 feed_dict={
                                     self.x: X,
                                     self.y: Y
                                 })
        
        return loss, accuracy

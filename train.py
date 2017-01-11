"""
Author: Vivek Ramanujan

Code for training the chess evaluation network.
"""

from models import Evaluation
from tqdm import tqdm
import data
import tensorflow as tf
import datetime
import os


flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_integer('batch_size', 32, 'Batch size for training')
flags.DEFINE_integer('image_channels', 1, 'Don\'t change this, for convolutional model')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for evaluation network')
flags.DEFINE_integer('image_size', 8, 'Don\'t change this')
flags.DEFINE_integer('hidden_size', 512, 'Hidden layer size for feedforward component')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs')
flags.DEFINE_string('dataset', '2015', 'Dataset to train on.\
                     Must be in datasets/pgns with a pgn extension')
flags.DEFINE_bool('overwrite_datasets', False, 'Whether or not to overwrite epd/csv')
flags.DEFINE_bool('verbose_mode', True, 'Verbose output during training')
flags.DEFINE_integer('embedding_size', 20, 'Embedding size for embedding feedforward model')
flags.DEFINE_bool('use_embedding', True, 'Whether or not to use embedding model')
flags.DEFINE_integer('save_step', 10000, 'Number of batches to save after')
flags.DEFINE_integer('display_step', 1, 'Number of epochs to display after')
flags.DEFINE_string('log_dir', 'logs/', 'Where to log checkpoints and summaries')
flags.DEFINE_bool('logging', True, 'Whether or not to enable logging summaries (checkpointing always on)')
flags.DEFINE_integer('num_eval_batches', 10, 'Number of batches to compare against for evaluation')


def main(_):
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d-%H%M")
    base_dir = os.path.join(FLAGS.log_dir, time)
    log_dir = os.path.join(base_dir, 'checkpoints/')
    summary_dir = os.path.join(base_dir, 'summaries/')

    csv_path = data.generate_csv(FLAGS.dataset, overwrite=FLAGS.overwrite_datasets,
                                                verbose=FLAGS.verbose_mode)
    
    dataset = data.generate_datasets(csv_path, FLAGS.batch_size,\
                                    overwrite=FLAGS.overwrite_datasets)

    print "Creating model..."
    eval_model = None

    if not FLAGS.use_embedding:
        eval_model = Evaluation.EvaluationNetworkConv()
    else:
        eval_model = Evaluation.EvaluationNetworkFC()

    saver = tf.train.Saver(tf.all_variables())

    print "Training..."

    print "Making logging directories..."
    tf.gfile.MakeDirs(log_dir)
    tf.gfile.MakeDirs(summary_dir)

    # Logging operations
    train_writer = tf.summary.FileWriter(os.path.join(summary_dir, 'train/'), eval_model.sess.graph)
    num_samples = dataset.train.num_examples

    total_cost = 0.0
    total_accuracy = 0.0
    for step in tqdm(xrange(int(FLAGS.num_epochs * num_samples / FLAGS.batch_size))):
        batch_x, batch_y = dataset.train.next_batch()
        loss, summary, accuracy = eval_model.partial_fit_step(batch_x, batch_y)
        total_cost += loss
        total_accuracy += accuracy

        if step % FLAGS.save_step == 0:
            # Printing summary
            train_writer.add_summary(summary, step)
            print "=" * 80
            print "Epoch: %d (step %d)" % (dataset.train.epoch, step)
            print "=" * 40
            print "Average Cost = %0.6f" % (float(total_cost) / (FLAGS.save_step))
            print "Training Accuracy = %0.4f" % (total_accuracy / FLAGS.save_step)
            # Getting validation accuracy
            total_accuracy = 0.0
            total_cost = 0.0
            for _ in xrange(FLAGS.num_eval_batches):
                loss, acc = eval_model.accuracy(*dataset.validation.next_batch())
                total_accuracy += acc
                total_cost += loss

            print "Validation Cost = %0.6f" % (total_cost/FLAGS.num_eval_batches)
            print "Validation Accuracy = %0.4f" % (total_accuracy/FLAGS.num_eval_batches)
            print "=" * 80
            saver.save(eval_model.sess, os.path.join(log_dir, 'eval_model.ckpt'), step)

            total_cost = 0.0
            total_accuracy = 0.0



if __name__ == '__main__':
    tf.app.run()
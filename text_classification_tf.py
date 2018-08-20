# -*- coding:utf-8 -*-
from __future__ import division
import random
import cv2
import os
import re
import time
import datetime
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import keras.backend as K
from scipy import interp
from time import sleep
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
FLAGS = tf.flags.FLAGS
from tensorflow.contrib import learn
# from attention_context import AttentionWithContext
random.seed(2018)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 150, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 300, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 300, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, sequence_length_left, sequence_length_right, num_classes, vocab_size, embedding_size, l2_reg_lambda=0.0):
        self.sequence_length_left = sequence_length_left
        self.sequence_length_right = sequence_length_right
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.l2_reg_lambda = l2_reg_lambda

        # Placeholders for input, output and dropout
        self.input_left = tf.placeholder(tf.int32, [None, self.sequence_length_left], name="input_left")
        self.input_right = tf.placeholder(tf.int32, [None, self.sequence_length_right], name="input_right")
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name="W_emb")
            self.embedded_chars_left = tf.nn.embedding_lookup(self.W, self.input_left)
            self.embedded_chars_expanded_left = tf.expand_dims(self.embedded_chars_left, -1)

            self.embedded_chars_right = tf.nn.embedding_lookup(self.W, self.input_right)
            self.embedded_chars_expanded_right = tf.expand_dims(self.embedded_chars_right, -1)
        print(self.embedded_chars_expanded_right)

        branch_am_cnn_left = self.branch_am_cnn(self.embedded_chars_expanded_left)
        branch_am_cnn_right = self.branch_am_cnn(self.embedded_chars_expanded_right)
        print(branch_am_cnn_left)
        num_filters_total = 128 + 128
        self.h_pool = tf.concat([branch_am_cnn_left, branch_am_cnn_right], 3)
        print(self.h_pool)
        self.h_pool_flat = tf.contrib.layers.flatten(self.h_pool)
        # self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print(self.h_pool_flat)

        # Add dropout
        with tf.name_scope("dropout1"):
            self.h_drop_1 = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            print(self.h_drop_1)

        with tf.name_scope("fc1"):
            W_fc1 = tf.get_variable("W_fc1", shape=[3072, 128], initializer=tf.contrib.layers.xavier_initializer())
            b_fc1 = tf.Variable(tf.constant(0.1, shape=[128]), name="b_fc1")
            # self.l2_loss_fc1 += tf.nn.l2_loss(W_fc1)
            # self.l2_loss_fc1 += tf.nn.l2_loss(b_fc1)
            self.z_fc1 = tf.nn.xw_plus_b(self.h_drop_1, W_fc1, b_fc1, name="scores_fc1")
            self.o_fc1 = tf.nn.relu(self.z_fc1, name="relu_fc1")

        # Add dropout
        with tf.name_scope("dropout2"):
            self.h_drop_2 = tf.nn.dropout(self.o_fc1, self.dropout_keep_prob)
            print(self.h_drop_2)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W_o = tf.get_variable("W_o", shape=[128, self.num_classes], initializer=tf.contrib.layers.xavier_initializer())
            b_o = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b_o")
            l2_loss += tf.nn.l2_loss(W_o)
            l2_loss += tf.nn.l2_loss(b_o)
            # self.scores_o = tf.reshape(self.h_drop_2, [-1, 128])
            self.scores_o = tf.nn.xw_plus_b(self.h_drop_2, W_o, b_o, name="scores_o")
            self.predictions = tf.argmax(self.scores_o, 1, name="predictions")
            print(self.scores_o)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores_o, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * l2_loss

    def branch_am_cnn(self, embedded_chars_expanded):
        filter_size_1, filter_size_2, filter_size_3 = 3, 3, 3
        num_filters_1, num_filters_2, num_filters_3 = 64, 128, 128
        with tf.name_scope("conv-maxpool-%s" % filter_size_1):
            # Convolution Layer
            filter_shape = [filter_size_1, self.embedding_size, 1, num_filters_1]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_1]), name="b")
            conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, self.embedding_size, 1], padding="SAME", name="conv1")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu1")

            # Maxpooling over the outputs
            # pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length_left - filter_size_1 + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
            pooled = tf.nn.max_pool(h, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool1")
            print(h)
            print(pooled)
            # pooled_outputs.append(pooled)

        with tf.name_scope("conv-maxpool-%s" % filter_size_2):
            # Convolution Layer
            # filter_shape = [filter_size_2, self.embedding_size, 1, num_filters_2]
            filter_shape = [filter_size_2, 1, num_filters_1, num_filters_2]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_2]), name="b")
            conv = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="SAME", name="conv2")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu2")
            # Maxpooling over the outputs
            # pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length_left - filter_size_2 + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
            pooled = tf.nn.max_pool(h, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool2")
            print(h)
            print(pooled)

        with tf.name_scope("conv-maxpool-%s" % filter_size_3):
            # Convolution Layer
            # filter_shape = [filter_size_3, self.embedding_size, 1, num_filters_3]
            filter_shape = [filter_size_3, 1, num_filters_2, num_filters_3]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters_3]), name="b")
            conv = tf.nn.conv2d(pooled, W, strides=[1, 1, 1, 1], padding="SAME", name="conv3")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu3")
            # Maxpooling over the outputs
            # pooled = tf.nn.max_pool(h, ksize=[1, self.sequence_length_left - filter_size_3 + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
            pooled = tf.nn.max_pool(h, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name="pool3")
            print(h)
            print(pooled)

        return pooled


class Train:
    def __init__(self):
        self.n_classes = 2
        self.nb_epoch = 10
        self.batch_size = 256 * 3
        self.val_split = 0.3
        self.MAX_ITEM_DESC_SEQ = 50

        self.train_data = 'E:/data/quora-duplicate/train.tsv'
        self.model_path = 'E:/data/quora-duplicate/model/'
        # self.train_data = 'H:/tb/project0/quora/quora_duplicate_questions.tsv'
        # self.model_path = 'H:/tb/project0/quora/model/'

    @staticmethod
    def evaluation(y_true, y_predict):
        accuracy = accuracy_score(y_true, y_predict)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_predict)
        print('accuracy:' + str(accuracy))
        print('precision:' + str(precision))
        print('recall:' + str(recall))
        print('f1:' + str(f1))

    def flatten(self, l):
        return [item for sublist in l for item in sublist]

    def data_cleaning(self, data):
        data['question1'] = data['question1'].str.lower()
        data['question1'].fillna(value="nan", inplace=True)
        data['question2'] = data['question2'].str.lower()
        data['question2'].fillna(value="nan", inplace=True)

        # f1 = lambda a: re.sub(r'(@.*? )', '', a)
        # f2 = lambda a: re.sub(r'(@.*?$)', '', a)
        # f3 = lambda a: re.sub(' +', ' ', a)
        # data['SentimentText'] = data['SentimentText'].apply(f1)
        # data['SentimentText'] = data['SentimentText'].apply(f2)
        # data['SentimentText'] = data['SentimentText'].apply(f3)

        # english_stopwords = stopwords.words('english')
        # list_senti = []
        # for row in data['SentimentText']:
        #     senti = [' '.join(a for a in row.split(' ') if a not in english_stopwords)]
        #     list_senti.append(senti)
        # data['SentimentText'] = list_senti

        return data

    def preprocessing(self, train_x, val_x):
        print("start preprocessing")
        raw_text = np.hstack([train_x['question1'], train_x['question2'], val_x['question1'], val_x['question2']])
        tok_raw = Tokenizer()
        tok_raw.fit_on_texts(raw_text)

        train_x['seq_question1'] = tok_raw.texts_to_sequences(train_x['question1'])
        train_x['seq_question2'] = tok_raw.texts_to_sequences(train_x['question2'])
        val_x['seq_question1'] = tok_raw.texts_to_sequences(val_x['question1'])
        val_x['seq_question2'] = tok_raw.texts_to_sequences(val_x['question2'])
        self.MAX_TEXT = np.unique(self.flatten(np.concatenate([train_x['seq_question1'], train_x['seq_question2'], val_x['seq_question1'], val_x['seq_question2']]))).shape[0] + 1

        train_Q1 = pad_sequences(train_x['seq_question1'], maxlen=self.MAX_ITEM_DESC_SEQ)
        train_Q2 = pad_sequences(train_x['seq_question2'], maxlen=self.MAX_ITEM_DESC_SEQ)
        val_Q1 = pad_sequences(val_x['seq_question1'], maxlen=self.MAX_ITEM_DESC_SEQ)
        val_Q2 = pad_sequences(val_x['seq_question2'], maxlen=self.MAX_ITEM_DESC_SEQ)
        return train_Q1, train_Q2, val_Q1, val_Q2

    def show_model_effect(self, history):
        """将训练过程中的评估指标变化可视化"""

        # summarize history for accuracy
        plt.plot(history.history["acc"])
        plt.plot(history.history["val_acc"])
        plt.title("Model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.savefig(self.model_path+"/Performance_accuracy.jpg")

        # summarize history for loss
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.savefig(self.model_path+"/Performance_loss.jpg")

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    def train(self, x_train_left, x_train_right, y_train, x_left_dev, x_right_dev, y_dev, vocab_processor):
        # Training
        # ==================================================

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement, log_device_placement=FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)
            # sess = tf.Session()
            with sess.as_default():
                cnn = TextCNN(sequence_length_left=x_train_left.shape[1],
                    sequence_length_right=x_train_right.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_dim)

                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
                grads_and_vars = optimizer.compute_gradients(cnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                # Keep track of gradient values and sparsity (optional)
                grad_summaries = []
                for g, v in grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)

                # Output directory for models and summaries
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                print("Writing to {}\n".format(out_dir))

                # Summaries for loss and accuracy
                loss_summary = tf.summary.scalar("loss", cnn.loss)
                acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

                # Train Summaries
                train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                # Dev summaries
                dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
                dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

                # Write vocabulary
                vocab_processor.save(os.path.join(out_dir, "vocab"))

                # Initialize all variables
                sess.run(tf.global_variables_initializer())

                def train_step(x_left_batch, x_right_batch, y_batch):
                    """
                    A single training step
                    """
                    feed_dict = {
                        cnn.input_left: x_left_batch,
                        cnn.input_right: x_right_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }
                    _, step, summaries, loss, accuracy = sess.run(
                        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    train_summary_writer.add_summary(summaries, step)

                def dev_step(x_left_batch, x_right_batch, y_batch, writer=None):
                    """
                    Evaluates model on a dev set
                    """
                    feed_dict = {
                        cnn.input_left: x_left_batch,
                        cnn.input_right: x_right_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: 1.0
                    }
                    step, summaries, loss, accuracy = sess.run([global_step, dev_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    if writer:
                        writer.add_summary(summaries, step)
                    return loss, accuracy

                # Generate batches
                batches = self.batch_iter(list(zip(x_train_left, x_train_right, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
                # Training loop. For each batch...
                for batch in batches:
                    x_left_batch, x_right_batch, y_batch = zip(*batch)
                    train_step(x_left_batch, x_right_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % FLAGS.evaluate_every == 0:
                        dev_batches = self.batch_iter(list(zip(x_left_dev, x_right_dev, y_dev)), FLAGS.batch_size, 1)
                        total_dev_correct = 0
                        print("\nEvaluation:")
                        for dev_batch in dev_batches:
                            x_left_dev_batch, x_right_dev_batch, y_dev_batch = zip(*dev_batch)
                            loss, dev_correct = dev_step(x_left_dev_batch, x_right_dev_batch, y_dev_batch)
                            total_dev_correct += dev_correct * len(y_dev_batch)
                            # dev_step(x_left_dev, x_right_dev, y_dev, writer=dev_summary_writer)
                        dev_accuracy = float(total_dev_correct) / len(y_dev)
                        print('Accuracy on dev set: {0}'.format(dev_accuracy))
                        print("Evaluation finished")
                    if current_step % FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def load_data_and_labels(self, positive_data_file, negative_data_file):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        # Load data from files
        positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
        positive_examples = [s.strip() for s in positive_examples]
        negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
        negative_examples = [s.strip() for s in negative_examples]
        # Split by words
        x_text = positive_examples + negative_examples
        x_text = [self.clean_str(sent) for sent in x_text]
        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        y = np.concatenate([positive_labels, negative_labels], 0)
        return [x_text, y]

    def preprocess(self):
        # 读取训练数据
        data = pd.read_csv(self.train_data, sep="\t", error_bad_lines=False)
        print(pd.value_counts(data['is_duplicate']))

        # 数据清洗
        data = self.data_cleaning(data)

        # Build vocabulary
        # max_document_length = max([len(x.split(" ")) for x in x_text])
        max_document_length = 100
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, min_frequency=3)
        vocab_processor.fit(data['question1'] + data['question2'])
        # x = np.array(list(vocab_processor.fit_transform(x_text)))
        x_left = np.array(list(vocab_processor.transform(data['question1'])))
        x_right = np.array(list(vocab_processor.transform(data['question2'])))
        y = to_categorical(data['is_duplicate'], num_classes=self.n_classes)
        print(y.shape)
        print(type(y))

        np.random.seed(2018)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_left_shuffled = x_left[shuffle_indices]
        x_right_shuffled = x_right[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
        x_left_train, x_left_dev = x_left_shuffled[:dev_sample_index], x_left_shuffled[dev_sample_index:]
        x_right_train, x_right_dev = x_right_shuffled[:dev_sample_index], x_right_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        del x_left, x_right, y, x_left_shuffled, x_right_shuffled, y_shuffled

        return x_left_train, x_right_train, y_train, x_left_dev, x_right_dev, y_dev, vocab_processor


if __name__ == '__main__':
    # 模型训练
    obj_train = Train()
    x_left_train, x_right_train, y_train, x_left_dev, x_right_dev, y_dev, vocab_processor = obj_train.preprocess()
    obj_train.train(x_left_train, x_right_train, y_train, x_left_dev, x_right_dev, y_dev, vocab_processor)


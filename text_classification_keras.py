# -*- coding:utf-8 -*-
from __future__ import division
import random
import cv2
import os
import re
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
from keras.layers.core import Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers import Flatten, Dense, Dropout, Lambda, Input, Embedding, Permute, RepeatVector, TimeDistributed
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, Convolution2D
from keras.layers import Conv1D, MaxPooling1D, ZeroPadding1D, AveragePooling1D
from keras.layers import LSTM, GRU, Bidirectional
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D, GlobalAveragePooling1D
from keras.layers import Activation, initializers
from keras.layers import concatenate, merge
from keras.optimizers import Nadam, Adadelta, Adam
from keras.regularizers import l2
# from attention_context import AttentionWithContext
random.seed(2018)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class TextModel:
    def __init__(self, N_CLASSES, MAX_TEXT, MAX_ITEM_DESC_SEQ):
        self.N_CLASSES = N_CLASSES
        self.MAX_TEXT = MAX_TEXT
        self.MAX_ITEM_DESC_SEQ = MAX_ITEM_DESC_SEQ
        self.encoder_units = 16
        self.decoder_units = 16
        self.emb_size = 40
        self.lr = 0.001
        self.dropout = 0.2

    @staticmethod
    def conv1d_bn(x, filters, width, padding='same', strides=1):
        """添加一个Conv+BN+ReLU组成的层"""
        if K.image_data_format() == 'channels_first':
            bn_axis = 1
        else:
            bn_axis = 3
        x = Conv1D(filters, width, strides=strides, padding=padding)(x)
        # x = BatchNormalization(axis=bn_axis, scale=False)(x)
        x = Activation('relu')(x)
        return x

    def branch_cnn(self, emb):
        """
        模仿googlenet的inception v2的结构，但nlp不适合太复杂的conv结构，因此做了简化
        """
        m_cnn_1 = self.conv1d_bn(emb, 64, 3, padding='same')
        m_cnn_2 = self.conv1d_bn(emb, 32, 3, padding='same')
        m_cnn_2 = self.conv1d_bn(m_cnn_2, 128, 3, padding='same')
        m_cnn_3 = self.conv1d_bn(emb, 128, 3, padding='same')
        m_cnn_3 = self.conv1d_bn(m_cnn_3, 64, 1, padding='same')
        m_cnn = concatenate([m_cnn_1, m_cnn_2, m_cnn_3])
        m_cnn = MaxPooling1D(pool_size=2, padding='valid')(m_cnn)
        m_cnn = self.conv1d_bn(m_cnn, 64, 3, padding='same')
        m_cnn = MaxPooling1D(pool_size=2, padding='valid')(m_cnn)
        m_cnn = Flatten()(m_cnn)
        return m_cnn

    # def branch_bilstm_am(self, emb):
    #     """
    #     真实使用的attention model
    #     """
    #     # m_lstm = LSTM(self.encoder_units, return_sequences=True, trainable=True)(emb)
    #     m_lstm = Bidirectional(LSTM(self.encoder_units, return_sequences=True, trainable=True))(emb)
    #     attention = AttentionWithContext()(m_lstm)
    #
    #     return attention

    def branch_cnn_am1(self, q1, q2, X_train_q1, X_train_q2):
        # 将输出映射到Embedding层，此处共享权重矩阵
        emb_layer = Embedding(self.MAX_TEXT, self.emb_size, trainable=True)
        emb_q1 = emb_layer(q1)
        emb_q2 = emb_layer(q2)

        # 计算attention相似度矩阵， 并根据emb_layer进行转换
        match_score = self.MatchScore(emb_q1, emb_q2, mode='cos')
        attention_left = TimeDistributed(Dense(self.emb_size, activation="tanh"), input_shape=(X_train_q1.shape[1], X_train_q2.shape[1]))(match_score)
        match_score_t = Permute((2, 1))(match_score)
        attention_right = TimeDistributed(Dense(self.emb_size, activation="tanh"), input_shape=(X_train_q2.shape[1], X_train_q1.shape[1]))(match_score_t)

        # 将emb layer从两维扩充为三维，因为需要叠加attention layer叠加 channel
        left_reshape = Reshape((1, attention_left._keras_shape[1], attention_left._keras_shape[2]))
        attention_left = left_reshape(attention_left)
        emb_q1 = left_reshape(emb_q1)

        right_reshape = Reshape((1, attention_right._keras_shape[1], attention_right._keras_shape[2]))
        attention_right = right_reshape(attention_right)
        emb_q2 = right_reshape(emb_q2)

        # 将emb layer与attention layer叠加
        emb_q1 = merge([emb_q1, attention_left], mode="concat", concat_axis=1)
        emb_q2 = merge([emb_q2, attention_right], mode="concat", concat_axis=1)

        # 为了下一步的wide convolution先做zero padding
        left_embed_padded = ZeroPadding2D((int(3 / 2), 0))(emb_q1)
        right_embed_padded = ZeroPadding2D((int(3 / 2), 0))(emb_q2)

        # 第一层当做image来做，做2d的convolution
        conv_left = Conv2D(filters=64, kernel_size=(3, self.emb_size), activation="tanh", padding="valid")(left_embed_padded)
        conv_left = (Reshape((conv_left._keras_shape[1], conv_left._keras_shape[2])))(conv_left)
        conv_left = AveragePooling1D(pool_size=3, strides=1, padding='same')(conv_left)

        # 连接正常的text的1d convolution
        conv_left = Conv1D(128, 3, strides=1, padding='valid')(conv_left)
        conv_left = Activation('relu')(conv_left)
        conv_left = MaxPooling1D(pool_size=2)(conv_left)
        conv_left = Dropout(0.2)(conv_left)

        conv_left = Conv1D(32, 3, strides=1, padding='valid')(conv_left)
        conv_left = Activation('relu')(conv_left)
        conv_left = MaxPooling1D(pool_size=2)(conv_left)

        # conv_right以相同的方式进行
        conv_right = Conv2D(filters=64, kernel_size=(3, self.emb_size), activation="tanh", padding="valid")(right_embed_padded)
        conv_right = (Reshape((conv_right._keras_shape[1], conv_right._keras_shape[2])))(conv_right)
        conv_right = AveragePooling1D(pool_size=3, strides=1, padding='same')(conv_right)

        conv_right = Conv1D(128, 3, strides=1, padding='valid', activation='relu')(conv_right)
        conv_right = MaxPooling1D(pool_size=2)(conv_right)
        conv_right = Dropout(0.2)(conv_right)

        conv_right = Conv1D(32, 3, strides=1, padding='valid', activation='relu')(conv_right)
        conv_right = MaxPooling1D(pool_size=2)(conv_right)

        # 将left和right进行concatenate
        cnn = concatenate([conv_left, conv_right])
        return cnn

    def build_cnn_lstm_am(self, X_train):
        sentimenttext = Input(shape=[X_train.shape[1]], name="seq_sentimenttext")
        emb_sentimenttext = Embedding(self.MAX_TEXT, self.emb_size, trainable=True)(sentimenttext)

        m_cnn = self.branch_cnn(emb_sentimenttext)
        m_lstm_am = self.branch_bilstm_am(emb_sentimenttext)
        m_sent_representation = concatenate([m_cnn, m_lstm_am])

        fc = Dense(128, activation='relu')(m_sent_representation)
        fc = Dropout(0.2)(fc)
        fc = Dense(64, activation='relu')(fc)
        fc = Dropout(0.2)(fc)

        output = Dense(self.N_CLASSES, activation='softmax')(fc)

        model = Model([sentimenttext], output)
        print(model.summary())
        return model

    def compute_euclidean_match_score(self, l_r):
        l, r = l_r
        denominator = 1. + K.sqrt(
            -2 * K.batch_dot(l, r, axes=[2, 2]) +
            K.expand_dims(K.sum(K.square(l), axis=2), 2) +
            K.expand_dims(K.sum(K.square(r), axis=2), 1)
        )
        denominator = K.maximum(denominator, K.epsilon())
        return 1. / denominator

    def compute_cos_match_score(self, l_r):
        l, r = l_r
        return K.batch_dot(
            K.l2_normalize(l, axis=-1),
            K.l2_normalize(r, axis=-1),
            axes=[2, 2]
        )

    def MatchScore(self, l, r, mode="euclidean"):
        if mode == "euclidean":
            return merge(
                [l, r],
                mode=self.compute_euclidean_match_score,
                output_shape=lambda shapes: (None, shapes[0][1], shapes[1][1])
            )
        elif mode == "cos":
            return merge(
                [l, r],
                mode=self.compute_cos_match_score,
                output_shape=lambda shapes: (None, shapes[0][1], shapes[1][1])
            )
        elif mode == "dot":
            return merge([l, r], mode="dot")
        else:
            raise ValueError("Unknown match score mode %s" % mode)

    def build_cnn_am(self, X_train_q1, X_train_q2):
        q1 = Input(shape=[X_train_q1.shape[1]], name="seq_question1")
        q2 = Input(shape=[X_train_q2.shape[1]], name="seq_question2")
        # conv_left = emb_q1
        # conv_right = emb_q2

        cnn = self.branch_cnn_am1(q1, q2, X_train_q1, X_train_q2)

        # cnn1 = self.branch_cnn(conv_left)
        # cnn2 = self.branch_cnn(conv_right)
        # cnn = concatenate([cnn1, cnn2])

        cnn = Flatten()(cnn)

        fc = Dense(128, activation='relu')(cnn)
        fc = Dropout(0.2)(fc)
        fc = Dense(64, activation='relu')(fc)
        fc = Dropout(0.2)(fc)

        output = Dense(self.N_CLASSES, activation='softmax')(fc)

        model = Model([q1, q2], output)
        print(model.summary())
        return model

    def build_cnn(self, X_train_q1, X_train_q2):
        emb_layer = Embedding(self.MAX_TEXT, self.emb_size, trainable=True)
        q1 = Input(shape=[X_train_q1.shape[1]], name="seq_question1")
        emb_q1 = emb_layer(q1)
        q2 = Input(shape=[X_train_q2.shape[1]], name="seq_question2")
        emb_q2 = emb_layer(q2)

        cnn1 = self.branch_cnn(emb_q1)
        cnn2 = self.branch_cnn(emb_q2)

        cnn = concatenate([cnn1, cnn2])
        fc = Dense(128, activation='relu')(cnn)
        fc = Dropout(0.4)(fc)
        fc = Dense(64, activation='relu')(fc)
        fc = Dropout(0.2)(fc)

        output = Dense(self.N_CLASSES, activation='softmax')(fc)

        model = Model([q1, q2], output)
        print(model.summary())
        return model

    def compile(self, model):
        model.compile(optimizer=Nadam(lr=self.lr), loss='binary_crossentropy', metrics=['accuracy'])
        # model.compile(optimizer='Adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
        return model


class Train:
    def __init__(self):
        self.n_classes = 2
        self.nb_epoch = 10
        self.batch_size = 256 * 3
        self.val_split = 0.3
        self.MAX_ITEM_DESC_SEQ = 50

        self.train_data = 'E:/data/quora-duplicate/train.tsv'
        self.model_path = 'E:/data/quora-duplicate/model/'

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

    def process(self):
        # 读取训练数据
        data = pd.read_csv(self.train_data, sep="\t", error_bad_lines=False)
        print(pd.value_counts(data['is_duplicate']))

        # 数据清洗
        data = self.data_cleaning(data)

        # 数据切分
        train_x, val_x, train_y, val_y = train_test_split(data[['question1', 'question2']],
                                                          data['is_duplicate'], test_size=self.val_split, random_state=2018)

        # 数据格式转换
        train_Q1, train_Q2, val_Q1, val_Q2 = self.preprocessing(train_x, val_x)

        # 将label转换成categorical_crossentropy支持的格式
        train_y = to_categorical(train_y, num_classes=self.n_classes)
        val_y = to_categorical(val_y, num_classes=self.n_classes)

        # 构建模型并编译
        model_obj = TextModel(N_CLASSES=self.n_classes, MAX_TEXT=self.MAX_TEXT, MAX_ITEM_DESC_SEQ=self.MAX_ITEM_DESC_SEQ)
        # text_model = model_obj.build_cnn_lstm_am(train_X)
        text_model = model_obj.build_cnn_am(train_Q1, train_Q2)
        # text_model = model_obj.build_cnn(train_Q1, train_Q2)
        text_model = model_obj.compile(text_model)

        sleep(5)

        # 添加Checkpoint
        model_info = "/cnn_bilstm_am_model_classNum2_"
        epoch_info = 'model-ep{epoch:03d}-acc{acc:.3f}-val_acc{val_acc:.3f}.h5'
        ckpt_fn = self.model_path + model_info + epoch_info
        ckpt = ModelCheckpoint(filepath=ckpt_fn, monitor='val_acc', save_best_only=False, mode='max')
        print(ckpt_fn)

        # 添加早停机制
        early_stopping = EarlyStopping(monitor='val_acc', patience=2, verbose=1)

        # 训练model，加入一些监控机制
        history = text_model.fit([train_Q1, train_Q2], train_y, epochs=self.nb_epoch, batch_size=self.batch_size,
                                 validation_data=([val_Q1, val_Q2], val_y), callbacks=[ckpt, early_stopping], verbose=1)

        # 存储模型（已在callback中存储）
        # text_model.save(self.model_path)

        # 展示训练集和验证集的评估指标变化
        self.show_model_effect(history)

        # 验证集上的效果评估
        y_predict = text_model.predict([val_Q1, val_Q2], batch_size=self.batch_size, verbose=0)
        print(y_predict)
        y_predict = np.argmax(y_predict, axis=1) + 1
        print(y_predict)
        y_true = np.argmax(val_y, axis=1) + 1
        self.evaluation(y_true, y_predict)


if __name__ == '__main__':
    # 模型训练
    obj_train = Train()
    obj_train.process()


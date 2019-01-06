#!/usr/bin/env python3

import os
import sys
from keras.models import Model
from keras.layers import Dense, Input, Flatten, MaxPooling1D, dot, Conv1D
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
from keras.layers.embeddings import Embedding
from math import ceil
import tensorflow as tf

import data_dense
from cfg import load


class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        self.val_log_dir = os.path.join(log_dir, 'validation')
        training_log_dir = os.path.join(log_dir, 'training')
        self.val_writer = None
        super().__init__(training_log_dir, **kwargs)

    def set_model(self, model):
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        logs = {k: v for k, v in logs.items() if k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


def train(properties_path='props.yml', io_argument_path='io_args.yml'):
    props = load(properties_path)
    args = load(io_argument_path)

    # Read vocabularies
    src_f_name = args['src_train']
    trg_f_name = args['trg_train']
    vs = data_dense.read_vocabularies(args['model_name'] + "-vocab.pickle",
                                      src_f_name,
                                      trg_f_name,
                                      False,
                                      props['ngram_length'])
    vs.trainable = False

    # Inputs: list of one Input per N-gram size
    src_inp = Input(shape=(props['max_sent_length'],),
                    name="source_ngrams_{N}".format(N=props['ngram_length'][0]),
                    dtype="int32")

    trg_inp = Input(shape=(props['max_sent_length'],),
                    name="target_ngrams_{N}".format(N=props['ngram_length'][0]),
                    dtype="int32")

    # Embeddings: list of one Embedding per input
    src_emb = Embedding(len(vs.source_ngrams[props['ngram_length'][0]]),
                        props['feature_count'],
                        input_length=props['max_sent_length'],
                        name="source_embedding_{N}".format(N=props['ngram_length'][0]))(src_inp)

    trg_emb = Embedding(len(vs.target_ngrams[props['ngram_length'][0]]),
                        props['feature_count'],
                        input_length=props['max_sent_length'],
                        name="target_embedding_{N}".format(N=props['ngram_length'][0]))(trg_inp)

    # Conv
    src_conv_out = Conv1D(props['feature_count'], (5,), padding='same', activation='relu')(src_emb)
    trg_conv_out = Conv1D(props['feature_count'], (5,), padding='same', activation='relu')(trg_emb)

    src_maxpool_out = MaxPooling1D(pool_size=props['max_sent_length'])(src_conv_out)
    trg_maxpool_out = MaxPooling1D(pool_size=props['max_sent_length'])(trg_conv_out)

    src_flat_out = Flatten()(src_maxpool_out)
    trg_flat_out = Flatten()(trg_maxpool_out)

    # yet one dense
    src_dense_out = Dense(props['gru_width'], name="source_dense")(src_flat_out)
    trg_dense_out = Dense(props['gru_width'], name="target_dense")(trg_flat_out)

    # ...and cosine between the source and target side
    merged_out = dot([src_dense_out, trg_dense_out], axes=1, normalize=True)

    # classification
    s_out = Dense(1, activation='sigmoid', name='classification_layer')(merged_out)

    model = Model(inputs=[src_inp, trg_inp], outputs=s_out)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    train_inf_iter = data_dense.InfiniteDataIterator(src_f_name, trg_f_name)
    train_batch_iter = data_dense.fill_batch(props['minibatch_size'],
                                             props['max_sent_length'],
                                             vs,
                                             train_inf_iter,
                                             props['ngram_length'])

    # dev iter
    dev_inf_iter = data_dense.InfiniteDataIterator(args['src_devel'], args['trg_devel'])
    dev_batch_iter = data_dense.fill_batch(props['minibatch_size'],
                                           props['max_sent_length'],
                                           vs,
                                           dev_inf_iter,
                                           props['ngram_length'])

    # save model json
    model_json = model.to_json()

    with open('{}.json'.format(args['model_name']), "w") as json_file:
        json_file.write(model_json)

    # callback to save weights after each epoch
    save_cb = ModelCheckpoint(filepath=args['model_name'] + '.{epoch:02d}.h5',
                              monitor='val_loss',
                              verbose=1,
                              save_best_only=False,
                              mode='auto')

    early_stop = EarlyStopping(monitor='val_loss',
                               patience=5)

    csv_logger = CSVLogger('./logs/train.log', append=True, separator=';')

    # tensor_board = TensorBoard(log_dir='./graph', write_graph=True, write_images=True)
    # callbacks = [save_cb, early_stop, csv_logger, tensor_board]

    callbacks = [save_cb, early_stop, csv_logger, TrainValTensorBoard(write_graph=False)]

    # steps per epoch or validation steps equal samples in train or devel dataset / batch size, e.g. 2700 / 200 = 14
    steps_per_epoch = ceil(len(train_inf_iter.data) / props['minibatch_size']) * 10
    val_steps = ceil(len(dev_inf_iter.data) / props['minibatch_size']) * 10

    model.fit_generator(train_batch_iter,
                        steps_per_epoch,
                        props['epochs'],
                        callbacks=callbacks,
                        validation_data=dev_batch_iter,
                        validation_steps=val_steps)


if __name__ == "__main__":
    sys.exit(train(*sys.argv[1:]))

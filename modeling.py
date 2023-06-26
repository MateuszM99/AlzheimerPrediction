# ResNet-50, ResNet-101, VGG16, DenseNet-121, Inception-v3, \
import numpy as np
import streamlit as st
import keras.backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import InputLayer, BatchNormalization, Dropout, Flatten, Dense, Activation, MaxPool2D, \
    Conv2D


def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


class ModelBuilder:
    def __init__(self, base_model, train, test, val):
        self.model = None
        self.model_history = None
        self.base_model = base_model
        self.train = train
        self.test = test
        self.val = val

    def build_model(self):
        for layer in self.base_model.layers:
            layer.trainable = False
        model = Sequential()
        model.add(self.base_model)
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(2048, kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1024, kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4, activation='softmax'))
        model.summary()
        self.model = model

    def train_model(self):
        OPT = tf.keras.optimizers.Adam(learning_rate=0.001)
        METRICS = [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            f1_score]

        self.model.compile(loss='categorical_crossentropy',
                           metrics=METRICS,
                           optimizer=OPT)

        earlystopping = EarlyStopping(monitor='val_auc',
                                      mode='max',
                                      patience=2,
                                      verbose=1)
        callback_list = [earlystopping]

        model_history = self.model.fit(self.train,
                                       validation_data=self.val,
                                       callbacks=callback_list,
                                       epochs=5,
                                       verbose=1)
        self.model_history = model_history

    def plot_training_metrics(self, y_actual, y_pred, classes):
        test_loss, acc, p, r, auc, f1 = self.model.evaluate(self.test, verbose=False)
        st.subheader("Model Results")
        st.write("Accuracy:", acc)
        st.write("Precision:", p)
        st.write("Recall:", r)
        st.write("AUC:", auc)
        st.write("F1 score:", f1)
        results = round(acc, 2) * 100
        st.text('Model Report:\n    ' + classification_report(y_actual, y_pred, target_names=classes))

        history_dict = self.model_history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        auc_values = history_dict['auc']
        val_auc_values = history_dict['val_auc']
        epochs = range(1, len(history_dict['auc']) + 1)
        max_auc = np.max(val_auc_values)
        min_loss = np.min(val_loss_values)

        fig1, ax1 = plt.subplots()
        plt.plot(epochs, loss_values, 'bo', label='Training loss')
        plt.plot(epochs, val_loss_values, 'cornflowerblue', label='Validation loss')
        plt.title('Validation Loss by Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.axhline(y=min_loss, color='darkslategray', linestyle='--')
        plt.legend()
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        plt.plot(epochs, auc_values, 'bo', label='Training AUC')
        plt.plot(epochs, val_auc_values, 'cornflowerblue', label='Validation AUC')
        plt.plot(epochs, [results / 100] * len(epochs), 'darkmagenta', linestyle='--', label='Test AUC')
        plt.title('Validation AUC by Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        plt.axhline(y=max_auc, color='darkslategray', linestyle='--')
        plt.legend()
        st.pyplot(fig2)

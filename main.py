from modeling import ModelBuilder
from preproccesor import DataPreprocessor
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import tensorflow as tf

CLASS_NAMES = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']


def show_images(generator, y_pred=None):
    labels = dict(zip([0, 1, 2, 3], CLASS_NAMES))
    x, y = generator.next()
    if y_pred is None:
        for i in range(6):
            fig, ax = plt.subplots()
            plt.imshow(x[i])
            plt.axis("off")
            plt.title("Class:{}".format(labels[np.argmax(y[i])]))
            st.pyplot(fig)
    else:
        for i in range(6):
            fig, ax = plt.subplots()
            plt.imshow(x[i])
            plt.axis("off")
            plt.title("Actual:{} \nPredicted:{}".format(labels[np.argmax(y[i])], labels[y_pred[i]]))
            st.pyplot(fig)


st.set_page_config()
st.title("Alzheimer prediction")
preprocessor = DataPreprocessor()
preprocessor.load_image_datasets()
show_images(preprocessor.train)
VGG19 = tf.keras.applications.VGG19(input_shape=(176, 208, 3),
                  include_top=False,
                 weights="imagenet")
ResNet50 = tf.keras.applications.ResNet50(input_shape=(176, 208, 3),
                  include_top=False,
                 weights="imagenet")
ResNet101 = tf.keras.applications.ResNet101(input_shape=(176, 208, 3),
                  include_top=False,
                 weights="imagenet")
DenseNet121 = tf.keras.applications.DenseNet121(input_shape=(176, 208, 3),
                  include_top=False,
                 weights="imagenet")
InceptionV3 = tf.keras.applications.InceptionV3(input_shape=(176, 208, 3),
                  include_top=False,
                 weights="imagenet")
base_models = [VGG19, ResNet50, ResNet101, DenseNet121, InceptionV3]
headers = ["VGG19", "ResNet50", "ResNet101", "DenseNet121", "InceptionV3"]

for i in range(len(base_models)):
    st.header(headers[i])
    model_class = ModelBuilder(base_models[i], preprocessor.train, preprocessor.test, preprocessor.val)
    model_class.build_model()
    model_class.train_model()
    y_prob = model_class.model.predict(preprocessor.test)
    y_pred = y_prob.argmax(axis=-1)
    y_actual = preprocessor.test.classes
    model_class.plot_training_metrics(y_actual, y_pred, CLASS_NAMES)
    show_images(preprocessor.test, y_pred)

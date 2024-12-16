"""
[model]
image_width    = 256
image_height   = 256
image_channels = 3

num_classes    = 1
base_filters   = 16
num_layers     = 8
dropout_rate   = 0.08
learning_rate  = 0.001
"""

import json
import os
import sys

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import shutil
import sys
import glob
import traceback
import random
import numpy as np
import cv2
import tensorflow as tf

from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Conv2D, Dropout, Conv2D, MaxPool2D

from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.activations import relu
from tensorflow.keras import Model
from tensorflow.keras.losses import  BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
#from tensorflow.keras.metrics import Mean
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import VGG16

from ConfigParser import ConfigParser

from EpochChangeCallback import EpochChangeCallback
from GrayScaleImageWriter import GrayScaleImageWriter

from losses import dice_coef, basnet_hybrid_loss, sensitivity, specificity
from losses import iou_coef, iou_loss, bce_iou_loss

"""
See: https://www.tensorflow.org/api_docs/python/tf/keras/metrics
Module: tf.keras.metrics
Functions
"""

"""
See also: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/engine/training.py
"""

MODEL  = "model"
TRAIN  = "train"
INFER  = "infer"
EVAL = "eval"
BEST_MODEL_FILE = "best_model.keras"

import csv

class TensorflowUNet_VGG16:

  def __init__(self, config_file):
    self.set_seed()

    self.config    = ConfigParser(config_file)
    image_height   = self.config.get(MODEL, "image_height")
    image_width    = self.config.get(MODEL, "image_width")
    image_channels = self.config.get(MODEL, "image_channels")
    num_classes    = self.config.get(MODEL, "num_classes")
    
    self.model     = self.create_vgg16_unet((image_height, image_width, image_channels), num_classes)
    
    learning_rate  = self.config.get(MODEL, "learning_rate")

    self.optimizer = Adam(learning_rate = learning_rate, 
         beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, 
         amsgrad=False)
    
    self.model_loaded = False

    # 2023/05/20 Modified to read loss and metrics from train_eval_infer.config file.
    binary_crossentropy = tf.keras.metrics.binary_crossentropy
    binary_accuracy     = tf.keras.metrics.binary_accuracy

    # Default loss and metrics functions
    self.loss    = binary_crossentropy
    self.metrics = [binary_accuracy]
    
    # Read a loss function name from our config file, and eval it.
    # loss = "binary_crossentropy"
    self.loss  = eval(self.config.get(MODEL, "loss"))

    # Read a list of metrics function names, ant eval each of the list,
    # metrics = ["binary_accuracy"]
    metrics  = self.config.get(MODEL, "metrics")
    self.metrics = []
    for metric in metrics:
      self.metrics.append(eval(metric))
    
    print("--- loss    {}".format(self.loss))
    print("--- metrics {}".format(self.metrics))
    
    self.model.compile(optimizer = self.optimizer, loss= self.loss, metrics = self.metrics)
   
    show_summary = self.config.get(MODEL, "show_summary")
    if show_summary:
      self.model.summary()

  def set_seed(self, seed=137):
    print("=== set seed {}".format(seed))
    random.seed    = seed
    np.random.seed = seed
    tf.random.set_seed(seed)

  def create_vgg16_unet(self, input_shape, num_classes):
      """ Input """
      inputs = Input(input_shape)
      """ Pre-trained VGG16 as Encoder """
      vgg16 = VGG16(include_top=False, weights="imagenet", input_tensor=inputs)
      # Extract specific layers for skip connections
      s1 = vgg16.get_layer("block1_conv2").output
      s2 = vgg16.get_layer("block2_conv2").output
      s3 = vgg16.get_layer("block3_conv3").output
      s4 = vgg16.get_layer("block4_conv3").output
      """ Bridge """
      b1 = vgg16.get_layer("block5_conv3").output
      """ Decoder (based on nnU-Net) """
      d1 = self.decoder_block(b1, s4, 512)
      d2 = self.decoder_block(d1, s3, 256)
      d3 = self.decoder_block(d2, s2, 128)
      d4 = self.decoder_block(d3, s1, 64)
      """ Output Layer """
      outputs = Conv2D(num_classes, (1, 1), padding="same", activation="sigmoid")(d4)
      model = Model(inputs, outputs, name="VGG16_nnU-Net")
      return model

  def decoder_block(self, inputs, skip_features, filters):
      """ Decoder block with skip connections """
      x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(inputs)
      x = concatenate([x, skip_features])
      x = Conv2D(filters, (3, 3), activation='relu', padding="same")(x)
      x = Dropout(0.3)(x)
      x = Conv2D(filters, (3, 3), activation='relu', padding="same")(x)
      return x

  def train(self, x_train, y_train):
    batch_size = self.config.get(TRAIN, "batch_size")
    epochs = self.config.get(TRAIN, "epochs")
    patience = self.config.get(TRAIN, "patience")
    eval_dir = self.config.get(TRAIN, "eval_dir")
    model_dir = self.config.get(TRAIN, "model_dir")
    metrics = ["accuracy", "val_accuracy"]
    try:
        metrics = self.config.get(TRAIN, "metrics")
    except:
        pass
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    weight_filepath = os.path.join(model_dir, BEST_MODEL_FILE)
    early_stopping = EarlyStopping(patience=patience, verbose=1)
    check_point = ModelCheckpoint(weight_filepath, verbose=1, save_best_only=True)
    epoch_change = EpochChangeCallback(eval_dir, metrics)
    history = self.model.fit(x_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=epochs, 
                             callbacks=[early_stopping, check_point, epoch_change], verbose=1)
    # Save training history as a JSON
    history_file = os.path.join(eval_dir, "train_history.json")
    with open(history_file, mode='w') as file:
        json.dump(history.history, file)


  # 2023/05/09
  def load_model(self) :
    rc = False
    if  not self.model_loaded:    
      model_dir  = self.config.get(TRAIN, "model_dir")
      weight_filepath = os.path.join(model_dir, BEST_MODEL_FILE)
      if os.path.exists(weight_filepath):
        self.model.load_weights(weight_filepath)
        self.model_loaded = True
        print("=== Loaded a weight_file {}".format(weight_filepath))
        rc = True
      else:
        message = "Not found a weight_file " + weight_filepath
        raise Exception(message)
    else:
      print("== Already loaded a weight file.")
    return rc

  # 2023/05/05 Added newly.    
  def infer(self, input_dir, output_dir, expand=True):
    writer       = GrayScaleImageWriter()
    # We are intereseted in png and jpg files.
    image_files  = glob.glob(input_dir + "/*.png")
    image_files += glob.glob(input_dir + "/*.jpg")
    image_files += glob.glob(input_dir + "/*.tif")
    #2023/05/15 Added *.bmp files
    image_files += glob.glob(input_dir + "/*.bmp")

    width        = self.config.get(MODEL, "image_width")
    height       = self.config.get(MODEL, "image_height")
    # 2023/05/24
    merged_dir   = None
    try:
      merged_dir = self.config.get(INFER, "merged_dir")
      if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)
      if not os.path.exists(merged_dir):
        os.makedirs(merged_dir)
    except:
      pass

    for image_file in image_files:
      basename = os.path.basename(image_file)
      name     = basename.split(".")[0]
      img      = cv2.imread(image_file, cv2.COLOR_BGR2RGB)
      h = img.shape[0]
      w = img.shape[1]
      # Any way, we have to resize input image to match the input size of our TensorflowUNet model.
      img         = cv2.resize(img, (width, height))
      predictions = self.predict([img], expand=expand)
      prediction  = predictions[0]
      image       = prediction[0]    
      # Resize the predicted image to be the original image size (w, h), and save it as a grayscale image.
      # Probably, this is a natural way for all humans. 
      mask = writer.save_resized(image, (w, h), output_dir, name)
      # 2023/05/24
      print("--- image_file {}".format(image_file))
      if merged_dir !=None:
        # Resize img to the original size (w, h)
        img   = cv2.resize(img, (w, h))
        img += mask
        merged_file = os.path.join(merged_dir, basename)
        cv2.imwrite(merged_file, img)

  def predict(self, images, expand=True):
    self.load_model()
    predictions = []
    for image in images:
      #print("=== Input image shape {}".format(image.shape))
      if expand:
        image = np.expand_dims(image, 0)
      pred = self.model.predict(image)
      predictions.append(pred)
    return predictions    


  def evaluate(self, x_test, y_test): 
    self.load_model()
    score = self.model.evaluate(x_test, y_test, verbose=1)
    print("Test loss    :{}".format(round(score[0], 4)))     
    print("Test accuracy:{}".format(round(score[1], 4)))

    y_pred_test = self.model.predict(x_test)
    dice_score = dice_coef(y_test, y_pred_test).numpy()
    print(f"Final Dice Coefficient on training data: {dice_score:.4f}")

    # Guardar resultados en un archivo CSV
    eval_dir = self.config.get(EVAL, "eval_dir")
    test_results_file = os.path.join(eval_dir, "test_results.csv")

    with open(test_results_file, mode='w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(["Metric", "Value"])
      writer.writerow(["Test Loss", round(score[0], 4)])
      writer.writerow(["Test Accuracy", round(score[1], 4)])
      writer.writerow(["Dice Coefficient", dice_score])

    # Leer resultados de entrenamiento y validación
    train_results_file = os.path.join(eval_dir, "train_results.csv")
    if os.path.exists(train_results_file):
        with open(train_results_file, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Saltar la cabecera
            train_results = list(reader)

        # Comparar resultados de entrenamiento y validación con los de prueba
        last_epoch = train_results[-1]
        train_loss = float(last_epoch[1])
        train_accuracy = float(last_epoch[2])
        val_loss = float(last_epoch[3])
        val_accuracy = float(last_epoch[4])

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")

        # Guardar comparación en un archivo CSV
        comparison_file = os.path.join(eval_dir, "comparison_results.csv")
        with open(comparison_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Train", "Validation", "Test"])
            writer.writerow(["Loss", train_loss, val_loss, round(score[0], 4)])
            writer.writerow(["Accuracy", train_accuracy, val_accuracy, round(score[1], 4)])

  def conv_block(self, input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = relu(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = relu(x)

    return x

if __name__ == "__main__":

  try:
    # Default config_file
    config_file    = "./train_eval_infer_vgg16.config"
    # You can specify config_file on your command line parammeter.
    if len(sys.argv) == 2:
      cfile = sys.argv[1]
      if not os.path.exists(cfile):
         raise Exception("Not found " + cfile)
      else:
        config_file = cfile

    config   = ConfigParser(config_file)

    width    = config.get(MODEL, "image_width")
    height   = config.get(MODEL, "image_height")
    channels = config.get(MODEL, "image_channels")

    if not (width == height and  height % 128 == 0 and width % 128 == 0):
      raise Exception("Image width should be a multiple of 128. For example 128, 256, 512")
    
    # Create a UNetMolde and compile
    model    = TensorflowUNet_VGG16(config_file)
    # Please download and install graphviz for your OS
    # https://www.graphviz.org/download/
    asset_dir = './asset'
    if not os.path.exists(asset_dir):
          os.makedirs(asset_dir)
    image_file = os.path.join(asset_dir, 'model_vgg16.png')
    tf.keras.utils.plot_model(model.model, to_file=image_file, show_shapes=True)

  except:
    traceback.print_exc()

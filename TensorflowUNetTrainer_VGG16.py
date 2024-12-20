import os

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"

import sys
import traceback

from ConfigParser import ConfigParser
from ImageMaskDataset import ImageMaskDataset
#from EpochChangeCallback import EpochChangeCallback

from TensorflowUNet_VGG16 import TensorflowUNet_VGG16

MODEL  = "model"
TRAIN  = "train"


if __name__ == "__main__":
  try:
    config_file    = "./train_eval_infer_vgg16.config"
    # You can specify config_file on your command line parammeter.
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    if not os.path.exists(config_file):
      raise Exception("Not found " + config_file)
  
    config   = ConfigParser(config_file)

    width    = config.get(MODEL, "image_width")
    height   = config.get(MODEL, "image_height")
    channels = config.get(MODEL, "image_channels")

    if not (width == height and  height % 128 == 0 and width % 128 == 0):
      raise Exception("Image width should be a multiple of 128. For example 128, 256, 512")
    
    # 1 Create train dataset
    dataset          = ImageMaskDataset(config_file)

    x_train, y_train = dataset.create(dataset=TRAIN)
    print(" len x_train {}".format(len(x_train)))
    print(" len y_train {}".format(len(y_train)))

    # 2 Create a UNetMolde and compile
    model          = TensorflowUNet_VGG16(config_file)

    # 3 Start training
    model.train(x_train, y_train)

  except:
    traceback.print_exc()
    

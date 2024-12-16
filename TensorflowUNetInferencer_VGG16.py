import os
import sys
import shutil


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"

import traceback

from ConfigParser import ConfigParser

from TensorflowUNet_VGG16 import TensorflowUNet_VGG16

MODEL  = "model"
TRAIN  = "train"
INFER  = "infer"


if __name__ == "__main__":
  try:
    config_file    = "./train_eval_infer_vgg16.config"
    # You can specify config_file on your command line parammeter.
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    
    if not os.path.exists(config_file):
      raise Exception("Not found " + config_file)

    config     = ConfigParser(config_file)

    width      = config.get(MODEL, "image_width")
    height     = config.get(MODEL, "image_height")
    channels   = config.get(MODEL, "image_channels")
    images_dir = config.get(INFER, "images_dir")
    output_dir = config.get(INFER, "output_dir")

    if not (width == height and  height % 128 == 0 and width % 128 == 0):
      raise Exception("Image width should be a multiple of 128. For example 128, 256, 512")
    
    # Create a UNetMolde and compile
    model     = TensorflowUNet_VGG16(config_file)
    
    if not os.path.exists(images_dir):
      raise Exception("Not found " + images_dir)

    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    model.infer(images_dir, output_dir, expand=True)

  except:
    traceback.print_exc()
    

# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# TensorflowUNetEvaluator.py
# 2023/05/10 to-arai
# 2023/06/13 Modified to use ImageMaskDataset instead of BrainTumorDataset

import os
import sys
import csv

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"

import traceback

from ConfigParser import ConfigParser
from ImageMaskDataset import ImageMaskDataset

from TensorflowUNet_VGG16 import TensorflowUNet_VGG16
from GrayScaleImageWriter import GrayScaleImageWriter

MODEL  = "model"
TRAIN  = "train"
EVAL   = "eval"


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
    if not (width == height and  height % 128 == 0 and width % 128 == 0):
      raise Exception("Image width should be a multiple of 128. For example 128, 256, 512")
    
    # 1 Create test dataset

    dataset          = ImageMaskDataset(config_file)

    original_data_path  = config.get(EVAL, "image_datapath")
    segmented_data_path = config.get(EVAL, "mask_datapath")
   
    x_test, y_test = dataset.create(dataset=EVAL)
    print(" len x_test {}".format(len(x_test)))
    print(" len y_test {}".format(len(y_test)))

    # 2 Create a UNetMolde and compile
    model          = TensorflowUNet_VGG16(config_file)

    # 3 Start training
    model.evaluate(x_test, y_test)

  except:
    traceback.print_exc()



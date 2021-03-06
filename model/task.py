"""Run a training job on Cloud ML Engine for a given use case.
Usage:
  model.task --job-dir <output_path> --image-path <image_path> [--resume <model_path>]


Options:
  -h --help     Show this screen.]
"""
from docopt import docopt
import os
import zipfile
from train_val_v3 import model

if __name__ == '__main__':
    arguments = docopt(__doc__)
    # Assign model variables to commandline arguments
    # stupid switch for gcloud for some reason
    image_path = arguments['<image_path>']
    output_path = arguments['<output_path>']
    model_path = arguments['<model_path>']
    print("Image path: " + image_path)
    print("Output path: " + output_path)
    # print("Model path: " + model_path)

    # Run the training job
    model_instance = model(image_path, output_path, model_path)
    model_instance.set_up_and_train()
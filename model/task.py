"""Run a training job on Cloud ML Engine for a given use case.
Usage:
  model.task --image-path <image-path> --job-dir <output_path>


Options:
  -h --help     Show this screen.]
"""
from docopt import docopt

from train_val_class_test import model

if __name__ == '__main__':
    arguments = docopt(__doc__)
    # Assign model variables to commandline arguments
    image_path = arguments['<image-path>']
    output_path = arguments['<output_path>']
    # Run the training job
    model_instance = model(image_path, output_path)
    model_instance.set_up_and_train()
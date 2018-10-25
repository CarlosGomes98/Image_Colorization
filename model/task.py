"""Run a training job on Cloud ML Engine for a given use case.
Usage:
  model.task --image_path <image_path> --output_path <output_path>


Options:
  -h --help     Show this screen.]
"""
from docopt import docopt

from model.train_val_v2 import model  # Your model.py file.

if __name__ == '__main__':
    arguments = docopt(__doc__)
    # Assign model variables to commandline arguments
    image_path = arguments['<train_data_paths>']
    output_path = arguments['<output_path>']
    # Run the training job
    model_instance = model(image_path, output_path)
    model_instance.train_and_evaluate()
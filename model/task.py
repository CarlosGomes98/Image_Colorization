"""Run a training job on Cloud ML Engine for a given use case.
Usage:
  model.task --job-dir <output_path> --image-path <image_path> 


Options:
  -h --help     Show this screen.]
"""
from docopt import docopt
import os
from model.train_val_class import model

if __name__ == '__main__':
    arguments = docopt(__doc__)
    # Assign model variables to commandline arguments
    # stupid switch for gcloud for some reason
    image_path = arguments['<output_path>']
    output_path = arguments['<image_path>']
    print("Image path: " + image_path)
    print("Output path: " + output_path)
    # os.system('gsutil -m cp -r ' + image_path + '/Train' + '/Train .')
    os.system('gsutil -m cp -r ' + image_path + '/Validation' + '/Validation .')
    os.system('gsutil cp ' + image_path + '/pts_in_hull.npy .')
    os.system('gsutil cp ' + image_path + '/rebalance.npy .')
    # Run the training job
    model_instance = model(image_path, output_path)
    model_instance.set_up_and_train()
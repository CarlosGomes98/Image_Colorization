from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow', 'keras', 'numpy', 'scikit-image', 'docopt']

setup(
    name='model.train_val_v2',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)
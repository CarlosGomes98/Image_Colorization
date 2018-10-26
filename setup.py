from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow==1.8.0', 'keras', 'numpy', 'scikit-image', 'docopt', 'requests==2.18.4']

setup(
    name='model.train_val_v2',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)
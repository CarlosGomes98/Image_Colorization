from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['keras', 'numpy', 'scikit-image', 'docopt', 'requests==2.18.4']

setup(
    name='model.train_val_class',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
)

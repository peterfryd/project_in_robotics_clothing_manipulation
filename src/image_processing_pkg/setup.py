from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'image_processing_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alex',
    maintainer_email='alexellegaard@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'image_to_base = image_processing_pkg.image_to_base:main',
            'get_pick_and_place_point = image_processing_pkg.get_pick_and_place_point:main'
        ],
    },
)

from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'clothing_ai_pkg'

# Get only files from data directory, not subdirectories
data_files_list = [f for f in glob('data/*') if os.path.isfile(f)]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'data'), data_files_list),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='peter',
    maintainer_email='frydensberg.peter@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    #tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'get_landmarks = clothing_ai_pkg.get_landmarks:main',
        ],
    },
)

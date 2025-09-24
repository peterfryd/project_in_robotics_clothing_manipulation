from setuptools import setup

package_name = 'vla_inference'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/srv', ['srv/ServiceSendToModel.srv']),
    ],
    install_requires=[
        'setuptools',
        'torch>=2.1.0',
        'torchvision>=0.16.0',
        'transformers>=4.40.0',
        'numpy>=1.26.0',
        'Pillow>=10.0.0',
        'opencv-python>=4.8.0',
        'typeguard>=2.17.1',
        'Cython>=0.30.0',
    ],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your@email.com',
    description='ROS2 package for running OpenVLA inference',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vla_inference = vla_inference.vla_inference:main',
        ],
    },
)

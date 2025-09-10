# project_in_robotics_clothing_manipulation



## Setup
Follow these steps in order to get started with manipulating clothes!

### 1) Install ROS2

Follow the guide for installing ROS2 Jazzy here [ROS2 Jazzy Install](https://docs.ros.org/en/jazzy/Installation/Alternatives/Ubuntu-Development-Setup.html).

### 2) Clone and setup submodules

Run the git clone command:

``` bash
git clone https://github.com/peterfryd/project_in_robotics_clothing_manipulation.git
``` 

Initialize the submodules of the repository:
``` bash
git submodule update --init --recursive
```

### 3) Source the ROS2 Functionality and Workspace

Run the following commands for sourcing the ROS workspace

``` bash
source /opt/ros/jazzy/setup.bash && source project_in_robotics_clothing_manipulation/install/local_setup.sh

``` 

### 4) Install ROS RealSense library

These steps are taken from [ROS2 Intel Realsense](https://github.com/IntelRealSense/realsense-ros).

Run the following command:
``` bash
sudo apt install ros-jazzy-librealsense2*
``` 

### 5) Setup python virtual environment and download python dependencies

Create a virtual environment parallel to this project folder:
``` bash
python3 -m venv env --system-site-packages
``` 
Source the environment:
``` bash
source env/bin/activate
``` 

Install requirements:
``` bash
pip install -r requirements.txt
``` 

### 6) Install ur_rtde c++ library and setup robot for remote control
Install ur_rtde:
``` bash
sudo add-apt-repository ppa:sdurobotics/ur-rtde
sudo apt-get update
sudo apt install librtde librtde-dev
```

On the teach pendent setup a static IP and enable the robot for remote control


### 7) Build the src code

in the project root run:
``` bash
python3 -m colcon build
``` 

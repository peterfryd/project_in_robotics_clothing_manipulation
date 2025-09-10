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

### 4) Install ROS RealSense SDK library
Follow the "Installation on Ubuntu" guide from this link: https://github.com/IntelRealSense/realsense-ros

I specifically did:
* Step 2 - Option 2: Install librealsense2
* step 3 - Option 2: Install from source

OBS: You might need to update you kernel headers (atleast i did using linux kernel 6.14), or change kernel version.

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

### 6) Build the src code

in the project root run:
``` bash
python3 -m colcon build
``` 
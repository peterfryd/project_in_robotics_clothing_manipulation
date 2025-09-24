# project_in_robotics_clothing_manipulation

## Robot password
eit2025

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

### 6) Install ur_rtde c++ library and setup robot for remote control
Install ur_rtde:
``` bash
sudo add-apt-repository ppa:sdurobotics/ur-rtde
sudo apt-get update
sudo apt install librtde librtde-dev
```

Install modbus:
``` bash
sudo apt install libmodbus-dev
```

### 7) Build the src code
in the project root run:
``` bash
python3 -m colcon build
``` 

### 8) Connect to the Robot
Connect your pc and the robot with an ethernet cable. Turn on the robot's control tablet ad go under settings -> system -> network. Set the network to have a static address:
- IP Address: 192.168.1.100
- Subnet Mask: 255.255.255.0

On your pc, go to advanced network configuration, edit the Ethernet-network and go to IPV4-Settings.
- Add the IP Address: 192.68.1.77
- Add the netmask: 255.255.255.0

You can check that the communication is working by running:

``` bash
ping 192.168.100.1
``` 

Set the robot to "External Control" on the ur tablet.

### 9) Setup Ucloud for model inference using Python Flask
Setup Ucloud to have a personal SSH key, and add it to the Ucloud job once it starts.

Clone the repository into a folder within UCloud.
``` bash
git clone https://github.com/peterfryd/project_in_robotics_clothing_manipulation.git
``` 

And checkout the branch for the Ucloud programmes.
``` bash
git checkout ucloud
``` 

Run the setup-script:
``` bash
. setup.sh
``` 

## Run the programme
Run the following commands:

Start the robot controller:
``` bash
ros2 run robot_controller_pkg robot_controller 
``` 
Publish a desired comman to the robot:
``` bash
ros2 topic pub /robot_cmd robot_controller_pkg/msg/RobotCmd "{delta_position: [0.05, 0.05, 0.05], delta_orientation: [0.1, 0.1, 0.1], delta_gripper: 0.5}"
``` 

And for the camera nodes:
``` bash
ros2 run realsense2_camera realsense2_camera_node
``` 

In Ucloud run the flask server:
``` bash
python3 web_server.py
``` 

On the local machine allow SSH-Tunneling:
``` bash
ssh -L 5000:localhost:80 ucloud@ssh.cloud.sdu.dk -p 2215
``` 

Run the node for sending an image to the flask server for inference:
``` bash
ros2 run vla_inference vla_inference <path_to_img (jpg)>
``` 

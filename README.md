# Project in Robotics - Clothing Manipulation

## Setup
Follow these steps in order to get started with folding clothes!

### 1) Install ROS2
Follow the guide for installing ROS2 Jazzy here [ROS2 Jazzy Install](https://docs.ros.org/en/jazzy/Installation/Alternatives/Ubuntu-Development-Setup.html).

### 2) Clone this repo and setup submodules
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

When mutliple options are available we this these specifically:
* Step 2 - Option 2: Install librealsense2
* step 3 - Option 2: Install from source

OBS: You might need to update you kernel headers (atleast we did using linux kernel 6.14), or change kernel version.

### 5) Build the src code
in the project root run:
``` bash
python3 -m colcon build
```
ros2 run system_integrator_pkg main2 "*"

### 6) Connect to the Robot
Connect your pc and the robot with an ethernet cable. On the robots control tablet go to settings -> system -> network. Set the network to have a static address:
- IP Address: 192.168.1.100
- Subnet Mask: 255.255.255.0

On your pc, edit the Ethernet-network IPV4-Settings.
- Set the IP Address: 192.168.1.104
- Set the netmask: 255.255.255.0

You can check that the communication is working by running:

``` bash
ping 192.168.1.100
```

### 7) Prepare the robot programme
On the robot, create a new program. This program will need to be running when folding so we recommend saving it.
In that program add a single "script"-node, and add the line "interpreter_mode()".


## Run the programme

First start the robot programme created in the last section.

Then run the launch file which start all neccesary services
``` bash
ros2 launch system_integrator_pkg main_system2.launch.py
``` 

In another terminal run the main program and specify which fold type to use. Use "#" for square fold and "*" for star fold.
``` bash
ros2 run system_integrator_pkg main2 "*"
```
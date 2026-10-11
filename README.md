# Autonomous T-Shirt Folding Using Landmark Detection and SIFT-Based Correction

This project has implemented autonomous robot T-shirt folding on a UR5 robot arm using a novel landmark detection with SIFT-based correction algorithm.

OBS: Go to the end of the document for a setup guide.

## Landmark detection of clothes using AI 

As the base of the landmark detector the ResNet18 CNN is used, which has been pretrained on the ImageNet-1K dataset.

### Finetuning
For finetuning the landmark detector the DeepFashion2 dataset is used. This dataset contains 801,000 annotated images of different types of clothes, including 82,000 images of T-shirts which will be used for the finetuning.
<img width="800" alt="image" src="https://github.com/user-attachments/assets/4e9354d8-9d95-4e92-a0f6-20b36e5dcfd7" />

### Performance
After finetuning, the landmark detector had an average NME of 0.022 across all the landmarks.
<img width="500" alt="image" src="https://github.com/user-attachments/assets/046192e9-9f07-428a-bd4d-a1f133227301" />
<br/><br/>
<img width="400" alt="image" src="https://github.com/user-attachments/assets/3831ef48-e53b-432f-8007-697808a2fc2e" />

### Error sources
There are multiple types of images in the dataset which the landmark detector has problems with, and because it has been trained on these, they likely result in worse performance and could be removed for future itereations of this project. 

Below are some examples.

<img width="400" alt="image" src="https://github.com/user-attachments/assets/505acc2b-cc13-47c2-8dc5-1d5eb47dc79a" />
<img width="400" alt="image" src="https://github.com/user-attachments/assets/387988d1-3b24-495e-ad14-f440396299b8" />
<img width="400" alt="image" src="https://github.com/user-attachments/assets/ac816323-c99f-479f-a4a8-5d22b865077d" />


## SIFT based landmark correction
It is not enough to have the aforementioned landmark detector because it only works on clothing which is relatively stretched out. Therefore this implementation takes an image of the T-shirt when it is stretched out and then compares that image to every step of the folding sequence.

The SIFT features of the newest image of the folding sequence are compared to the SIFT features of the image of the stretched out T-shirt to find the shared features. Then the N=12 nearest shared features to each landmark are used to transform the landmark. 
<img width="400" alt="image" src="https://github.com/user-attachments/assets/784dbea9-21b7-48d8-98aa-8eefa935743a" />
<br/><br/>
<img width="400" alt="image" src="https://github.com/user-attachments/assets/665aecb5-6f98-494b-a345-3255c930f96a" />


## Experiment and results
An experiment was conducted to test the successrate of the folding. This was done on two different T-shirts and two different folding sequences with 16 trials with each combination resulting in 64 total trials.
This test resulted in an average successrate of 78.1% 
<img width="400" alt="image" src="https://github.com/user-attachments/assets/05d03b60-cff7-4884-bfd7-0048be9d00bf" />

A Fishers exact test showed no statistical difference in the successrate between the fold types or T-shirts.

## Demonstration
TODO









# Setup guide
Follow these steps in order to get started with folding clothes!

## 1) Install ROS2
Follow the guide for installing ROS2 Jazzy here [ROS2 Jazzy Install](https://docs.ros.org/en/jazzy/Installation/Alternatives/Ubuntu-Development-Setup.html).

## 2) Clone this repo and setup submodules
Run the git clone command:

``` bash
git clone https://github.com/peterfryd/project_in_robotics_clothing_manipulation.git
``` 

Initialize the submodules of the repository:
``` bash
git submodule update --init --recursive
```

## 3) Source the ROS2 Functionality and Workspace
Run the following commands for sourcing the ROS workspace

``` bash
source /opt/ros/jazzy/setup.bash && source project_in_robotics_clothing_manipulation/install/local_setup.sh
``` 

## 4) Install ROS RealSense SDK library
Follow the "Installation on Ubuntu" guide from this link: https://github.com/IntelRealSense/realsense-ros

When mutliple options are available we this these specifically:
* Step 2 - Option 2: Install librealsense2
* step 3 - Option 2: Install from source

OBS: You might need to update you kernel headers (atleast we did using linux kernel 6.14), or change kernel version.

## 5) Build the src code
in the project root run:
``` bash
python3 -m colcon build
```
ros2 run system_integrator_pkg main2 "*"

## 6) Connect to the Robot
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

## 7) Prepare the robot programme
On the robot, create a new program. This program will need to be running when folding so we recommend saving it.
In that program add a single "script"-node, and add the line "interpreter_mode()".


# Run the programme

First start the robot programme created in the last section.

Then run the launch file which start all neccesary services
``` bash
ros2 launch system_integrator_pkg main_system2.launch.py
``` 

In another terminal run the main program and specify which fold type to use. Use "#" for square fold and "*" for star fold.
``` bash
ros2 run system_integrator_pkg main2 "*"
```

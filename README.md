# Auditory active sensing with the MiRo robot
### PGT dissertation by Yijing Shen (project code)

## About

This program shows how MiRo can localise sounds using the two microphones in its ears.  
The algorithm is an adapted version of the Jeffress model, with the added dynamic noise thresholding.

## Prerequisites
The code is packaged as a ROS 1 package, and requires ROS + MDK installed.

## Installation
Clone the repo into `~/mdk/catkin/src` and build with the following commands:

```shell
cd ~/mdk/catkin_ws/src
git clone https://github.com/MiRo-projects/miro_active_hearing
catkin build
source ~/.bashrc
```

## Running 
The program runs as a single ROS node, with the main file being `point_to_sound.py` that can be run with `rosrun`:

```bash
rosrun miro_active_hearing point_to_sound.py
```

## Notes
By default, the node only shows visualisation for the audio signals.  
To enable actuation on the real robot, comment out the lines **89** and **304** in the `point_to_sound.py`

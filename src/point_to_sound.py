#!/usr/bin/env python3

"""
MiRo orienting towards a sound
"""

import os
import numpy as np
import rospy
import miro2 as miro
import geometry_msgs
from node_detect_audio_engine import DetectAudioEngine
from std_msgs.msg import Int16MultiArray
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class AudioClient():
   
    def __init__(self):       
        #Microphone Parameters
        # Number of points to display
        self.x_len = 40000
        # number of microphones coming through on topic
        self.no_of_mics = 4

        #Generate figure for plotting mics
        self.fig = plt.figure()
        self.fig.suptitle("Microphones") # Give figure title

		#HEAD
        self.head_plot = self.fig.add_subplot(4,1,3)
        self.head_plot.set_ylim([-33000, 33000])
        self.head_plot.set_xlim([0, self.x_len])
        self.head_xs = np.arange(0, self.x_len)
        self.head_plot.set_xticklabels([])
        self.head_plot.set_yticks([])
        self.head_plot.grid(which="both", axis="x")
        self.head_plot.set_ylabel("Head", rotation=0, ha="right")
        self.head_ys = np.zeros(self.x_len)
        self.head_line, = self.head_plot.plot(self.head_xs, self.head_ys, linewidth=0.5, color="g")


        #LEFT EAR
        self.left_ear_plot = self.fig.add_subplot(4,1,1)
        self.left_ear_plot.set_ylim([-33000, 33000])
        self.left_ear_plot.set_xlim([0, self.x_len])
        self.left_ear_xs = np.arange(0, self.x_len)
        self.left_ear_plot.set_xticklabels([])
        self.left_ear_plot.set_yticks([])
        self.left_ear_plot.grid(which="both", axis="x")
        self.left_ear_plot.set_ylabel("Left Ear", rotation=0, ha="right")
        self.left_ear_ys = np.zeros(self.x_len)
        self.left_ear_line, = self.left_ear_plot.plot(self.left_ear_xs, self.left_ear_ys, linewidth=0.5, color="b")

        #RIGHT EAR
        self.right_ear_plot = self.fig.add_subplot(4,1,2)
        self.right_ear_plot.set_ylim([-33000, 33000])
        self.right_ear_plot.set_xlim([0, self.x_len])
        self.right_ear_xs = np.arange(0, self.x_len)
        self.right_ear_plot.set_xticklabels([])
        self.right_ear_plot.set_yticks([])
        self.right_ear_plot.grid(which="both", axis="x")
        self.right_ear_plot.set_ylabel("Right Ear", rotation=0, ha="right")
        self.right_ear_ys = np.zeros(self.x_len)
        self.right_ear_line, = self.right_ear_plot.plot(self.right_ear_xs, self.right_ear_ys, linewidth=0.5, color="r")

        #Tail
        self.tail_plot = self.fig.add_subplot(4,1,4)
        self.tail_plot.set_ylim([-33000, 33000])
        self.tail_plot.set_xlim([0, self.x_len])
        self.tail_xs = np.arange(0, self.x_len)
        self.tail_plot.set_yticks([])
        self.tail_plot.set_xlabel("Samples")
        self.tail_plot.grid(which="both", axis="x")
        self.tail_plot.set_ylabel("Tail", rotation=0, ha="right")
        self.tail_ys = np.zeros(self.x_len)
        self.tail_line, = self.tail_plot.plot(self.tail_xs, self.tail_ys, linewidth=0.5, color="c")

        self.ani = animation.FuncAnimation(self.fig, self.update_line, fargs=(self.left_ear_ys,self.right_ear_ys, self.head_ys, self.tail_ys,), init_func=self.animation_init, interval=10, blit=False)
        self.fig.subplots_adjust(hspace=0, wspace=0)

        self.input_mics = np.zeros((self.x_len, self.no_of_mics))
        #print(self.input_mics) 

        # which miro
        topic_base_name = "/" + os.getenv("MIRO_ROBOT_NAME")
       
        # subscribers
        self.sub_mics = rospy.Subscriber(topic_base_name + "/sensors/mics",
            Int16MultiArray, self.callback_mics, queue_size=1, tcp_nodelay=True)

        # publishers
        self.pub_push = rospy.Publisher(topic_base_name + "/core/mpg/push", miro.msg.push, queue_size=0)

        # prepare push message
        self.msg_push = miro.msg.push()
        self.msg_push.link = miro.constants.LINK_HEAD
        #self.msg_push.flags = miro.constants.PUSH_FLAG_VELOCITY
        #self.msg_push.flags = miro.constants.PUSH_FLAG_NO_TRANSLATION
        self.msg_push.flags = miro.constants.PUSH_FLAG_NO_TRANSLATION + miro.constants.PUSH_FLAG_VELOCITY

        # status flags
        self.audio_event = None
        self.orienting = False
        self.action_time = 1 #secs
        self.thresh = 0.05
        

    def callback_mics(self, data):
        self.audio_event = AudioEng.process_data(data.data)
        
        # data for display

        data = np.asarray(data.data)
        data = np.transpose(data.reshape((self.no_of_mics, 500)))
        #data = np.transpose(data.reshape((self.no_of_mics, 20000)))
        data = np.flipud(data)
        self.input_mics = np.vstack((data, self.input_mics[:self.x_len-500,:]))
        #print(self.audio_event) 

    # def loop(self):

    #     while not rospy.core.is_shutdown():
    #         #plt.show()
    #         if self.audio_event is None:
    #             continue
    #         if self.audio_event[0] is None:
    #             continue
    #         ae = self.audio_event[0]
    #         ae_head = self.audio_event[1]

    #         print("Azimuth: {:.2f}; Elevation: {:.2f}; Level : {:.2f}".format(ae.azim, ae.elev, ae.level))
    #         print("X: {:.2f}; Y: {:.2f}; Z : {:.2f}".format(ae_head.x, ae_head.y, ae_head.z))

    #         # if the event level is above the threshold than "push" towards it

    #         if ae.level >= 0.02:
    #             # send push
    #             self.msg_push.pushpos = geometry_msgs.msg.Vector3(
    #                 miro.constants.LOC_NOSE_TIP_X, 
    #                 miro.constants.LOC_NOSE_TIP_Y, 
    #                 miro.constants.LOC_NOSE_TIP_Z
    #             )
    #             self.msg_push.pushvec = geometry_msgs.msg.Vector3(
    #                 miro.constants.LOC_NOSE_TIP_X + ae_head.x,
    #                 miro.constants.LOC_NOSE_TIP_Y + ae_head.y,
    #                 miro.constants.LOC_NOSE_TIP_Z
    #             )
    #             self.pub_push.publish(self.msg_push)


    def loop_idea2(self):
        while not rospy.core.is_shutdown():
            # print(self.orienting)
            if self.orienting:
                self.pub_push.publish(self.msg_push)
                if rospy.Time.now() > self.start_time + rospy.Duration(0.5):
                    self.orienting = False
                # print("Azimuth: {:.2f}; Elevation: {:.2f}; Level : {:.2f}".format(ae.azim, ae.elev, ae.level))
                # print("X: {:.2f}; Y: {:.2f}; Z : {:.2f}".format(ae_head.x, ae_head.y, ae_head.z))

            else:
                if self.audio_event is None:
                    continue
                if self.audio_event[0] is None:
                    continue
                ae = self.audio_event[0]
                ae_head = self.audio_event[1]

                print("Azimuth: {:.2f}; Elevation: {:.2f}; Level : {:.2f}".format(ae.azim, ae.elev, ae.level))
                print("X: {:.2f}; Y: {:.2f}; Z : {:.2f}".format(ae_head.x, ae_head.y, ae_head.z))

                # if the event level is above the threshold than "push" towards it

                if ae.level >= self.thresh:
                    # send push
                    self.msg_push.pushpos = geometry_msgs.msg.Vector3(
                        miro.constants.LOC_NOSE_TIP_X, 
                        miro.constants.LOC_NOSE_TIP_Y, 
                        miro.constants.LOC_NOSE_TIP_Z
                    )
                    self.msg_push.pushvec = geometry_msgs.msg.Vector3(
                        miro.constants.LOC_NOSE_TIP_X + ae_head.x,
                        miro.constants.LOC_NOSE_TIP_Y + ae_head.y,
                        miro.constants.LOC_NOSE_TIP_Z
                    )
                    self.orienting = True
                    self.start_time = rospy.Time.now()

    def update_line(self, i, left_ear_ys, right_ear_ys, head_ys, tail_ys):
        #Flip buffer so that incoming data moves in from the right
        left_ear_data = np.flipud(self.input_mics[:, 0])
        right_ear_data = np.flipud(self.input_mics[:, 1])
        head_data = np.flipud(self.input_mics[:, 2])
        tail_data = np.flipud(self.input_mics[:, 3])

        #Append new buffer data to plotting data
        left_ear_ys = np.append(left_ear_ys, left_ear_data)
        right_ear_ys = np.append(right_ear_ys, right_ear_data)
        head_ys = np.append(head_ys, head_data)
        tail_ys = np.append(tail_ys, tail_data)

        #Remove old sample outside of plot
        left_ear_ys = left_ear_ys[-self.x_len:]
        right_ear_ys = right_ear_ys[-self.x_len:]
        head_ys = head_ys[-self.x_len:]
        tail_ys = tail_ys[-self.x_len:]

        #Set data to line
        self.left_ear_line.set_ydata(left_ear_ys)
        self.right_ear_line.set_ydata(right_ear_ys)
        self.head_line.set_ydata(head_ys)
        self.tail_line.set_ydata(tail_ys)

        #Return the line to be animated
        return self.left_ear_line, self.right_ear_line, self.head_line, self.tail_line,

    def animation_init(self):
        self.left_ear_line.set_ydata(np.zeros(self.x_len))
        self.right_ear_line.set_ydata(np.zeros(self.x_len))
        self.head_line.set_ydata(np.zeros(self.x_len))
        self.tail_line.set_ydata(np.zeros(self.x_len))
        return self.left_ear_line, self.right_ear_line, self.head_line, self.tail_line,


      

if __name__ == "__main__":

    rospy.init_node("point_to_sound", anonymous=True)
    AudioEng = DetectAudioEngine()
    main = AudioClient()
    #plt.show()
    main.loop_idea2()
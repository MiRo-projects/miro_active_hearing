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

class AudioClient():
   
    def __init__(self):

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
        self.action_time = 2 #secs

    def callback_mics(self, data):
        self.audio_event = AudioEng.process_data(data.data)

        #print(self.audio_event) 

    def loop(self):

        while not rospy.core.is_shutdown():
            if self.audio_event is None:
                continue
            if self.audio_event[0] is None:
                continue
            ae = self.audio_event[0]
            ae_head = self.audio_event[1]

            print("Azimuth: {:.2f}; Elevation: {:.2f}; Level : {:.2f}".format(ae.azim, ae.elev, ae.level))
            print("X: {:.2f}; Y: {:.2f}; Z : {:.2f}".format(ae_head.x, ae_head.y, ae_head.z))

            # if the event level is above the threshold than "push" towards it

            if ae.level >= 0.03:
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
                self.pub_push.publish(self.msg_push)


    # def loop_idea2(self):

    #     while not rospy.core.is_shutdown():
    #         print(self.orienting)
    #         if self.orienting:
    #             self.pub_push.publish(self.msg_push)
    #             if rospy.Time.now() > self.start_time + rospy.Duration(0.5):
    #                 self.orienting = False
    #         else:
    #             if self.audio_event is None:
    #                 continue
    #             if self.audio_event[0] is None:
    #                 continue
    #             ae = self.audio_event[0]
    #             ae_head = self.audio_event[1]

    #             print("Azimuth: {:.2f}; Elevation: {:.2f}; Level : {:.2f}".format(ae.azim, ae.elev, ae.level))
    #             print("X: {:.2f}; Y: {:.2f}; Z : {:.2f}".format(ae_head.x, ae_head.y, ae_head.z))

    #             # if the event level is above the threshold than "push" towards it

    #             if ae.level >= 0.02:
    #                 # send push
    #                 self.msg_push.pushpos = geometry_msgs.msg.Vector3(
    #                     miro.constants.LOC_NOSE_TIP_X, 
    #                     miro.constants.LOC_NOSE_TIP_Y, 
    #                     miro.constants.LOC_NOSE_TIP_Z
    #                 )
    #                 self.msg_push.pushvec = geometry_msgs.msg.Vector3(
    #                     miro.constants.LOC_NOSE_TIP_X + ae_head.x,
    #                     miro.constants.LOC_NOSE_TIP_Y + ae_head.y,
    #                     miro.constants.LOC_NOSE_TIP_Z
    #                 )
    #                 self.orienting = True
    #                 self.start_time = rospy.Time.now()
            

if __name__ == "__main__":

    rospy.init_node("point_to_sound", anonymous=True)
    AudioEng = DetectAudioEngine()
    main = AudioClient()
    main.loop()

    
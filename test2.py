# test2 library for jupyter notebooks.

# import necessary packages

import cv2
import numpy as np
from math import ceil
from scipy import misc
from matplotlib import pyplot as plt
from IPython.display import clear_output, HTML
from foscam import FoscamCamera, FOSCAM_SUCCESS

class IPCamera(object):

    def __init__(self, ip, port=88, user=None, password=None):
        self.mycam = FoscamCamera(ip, port, user, password, daemon=False, verbose=False)
        self.last_frame = []

    def get_frame(self, callback=None):
        try:
            ret, jpg = self.mycam.snap_picture_2()
            if ret is FOSCAM_SUCCESS:
                img_array = np.asarray(bytearray(jpg), dtype=np.uint8)
                frame = cv2.imdecode(img_array, 1)
                if callback:
                    frame = callback(frame)
                self.last_frame = frame
        except:
            pass
        return self.last_frame

    def release(self):
        pass

class Camera(object):

    def __init__(self, camera=0):
        self.cam = cv2.VideoCapture(camera)
        self.last_frame = []
        
    def get_frame(self, callback=None):
        _, frame = self.cam.read()
        if _:
            if callback:
                frame = callback(frame)
            self.last_frame = frame
        return self.last_frame
    
    def release(self):
        self.cam.release()

def show_frames(plots, cols=2, x=10, y=5):
    plt.figure(figsize=(x,y))
    i = 1
    c = len(plots)
    for plot in plots:
        ax = plt.subplot(ceil(c/2.), cols, i)
        try:
            plt.imshow(plot)
        except:
            pass
        plt.axis('off')
        i += 1
    plt.show()
    clear_output(wait=True)

face_cascade = cv2.CascadeClassifier('./data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./data/haarcascades/haarcascade_eye.xml')

def faces(color_frame, gr_frame):
    frame = color_frame
    faces = face_cascade.detectMultiScale(gr_frame, 1.3, 5)
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(color_frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gr = gr_frame[y:y+h, x:x+w]
        roi_color = color_frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gr)
        for (ex,ey,ew,eh) in eyes:
            frame = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return frame

def process(frame):
    frame = misc.imresize(frame, 25, 'bicubic')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = faces(frame, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return frame

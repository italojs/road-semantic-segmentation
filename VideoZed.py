import cv2
import helper
import pyzed.sl as sl
import numpy as np
from threading import Thread

class VideoZed:
    """
    Class that continuously read a frame from ZED using a dedicated thread.
    """

    def __init__(self, sess, image_shape, logits, keep_prob, input_image):
        self.frame = None
        self.stopped = False
        self.sess = sess
        self.image_shape = image_shape
        self.logits = logits
        self.keep_prob = keep_prob
        self.input_image = input_image

    def start(self):    
        Thread(target=self.run, args=()).start()
        return self

    def run(self):
        init = sl.InitParameters()
        cam = sl.Camera()
        if not cam.is_opened():
            print("Opening ZED Camera...")
        status = cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit()

        runtime = sl.RuntimeParameters()
        mat = sl.Mat()

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

        while not self.stopped: 
            err = cam.grab(runtime)
            if err == sl.ERROR_CODE.SUCCESS:
                cam.retrieve_image(mat, sl.VIEW.VIEW_LEFT)
                frameLocal = cv2.resize(mat.get_data(), (self.image_shape[1], self.image_shape[0]))
                
                # Transform a PNG frame to JPG, removing the last dimension.
                frameLocal = frameLocal[:,:,0:3]
                
                self.frame = np.array(helper.predict(self.sess, frameLocal,
                                                     self.input_image,
                                                     self.keep_prob,
                                                     self.logits,
                                                     self.image_shape))

    def stop(self):
        self.stopped = True
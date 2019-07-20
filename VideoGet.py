from threading import Thread
import cv2
import helper
import numpy as np

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src, sess, image_shape, logits, keep_prob, input_image):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        
        self.sess = sess
        self.image_shape = image_shape
        self.logits = logits
        self.keep_prob = keep_prob
        self.input_image = input_image

    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, frameLocal) = self.stream.read()
                frameLocal = cv2.resize(frameLocal, (self.image_shape[1], self.image_shape[0]))
                self.frame = np.array(helper.predict(self.sess, frameLocal, \ 
                                                     self.input_image, \ 
                                                     self.keep_prob, \ 
                                                     self.logits, \ 
                                                     self.image_shape))

    def stop(self):
        self.stopped = True
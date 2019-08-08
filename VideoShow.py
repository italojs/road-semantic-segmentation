from threading import Thread
import cv2

class VideoShow:
    """
    Class that continuously shows a frame using a dedicated thread.
    """

    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        # To resize the frame image uncomment the lines below
        # dim = (2560, 720)
        # dim = (1280, 720)
        while not self.stopped:
            # self.frame = cv2.resize(self.frame, dim, interpolation = cv2.INTER_LINEAR)
            # self.frame = cv2.resize(self.frame, dim, interpolation = cv2.INTER_CUBIC)
            cv2.imshow("Video", self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        self.stopped = True
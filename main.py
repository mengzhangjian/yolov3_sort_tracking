# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import glob
from pydarknet import Detector, Image

net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), bytes("cfg/yolov3.weights", encoding="utf-8"), 0,
                   bytes("cfg/coco.data", encoding="utf-8"))


label = open("cfg/coco_labels.txt").read().strip().split("\n") 
name = {k: v for k, v in enumerate(label)}
name_reverse = {v: k for k, v in name.items()}

def detect(img):
    """format yolo result"""

    dark_frame = Image(img)
    results = net.detect(dark_frame)
    del dark_frame
    id = -1
    dets = []
    for cats, score, bound in results:
        x, y, w, h = bound
        if cats.decode('utf-8') in name_reverse:
            id = name_reverse[cats.decode('utf-8')]
            left = int(x - w / 2)
            top = int(y - h / 2)
            right = int(x + w / 2)
            bottom = int(y + h /2)
            dets.append([left, top, right, bottom, float(score), id])
    return dets

files = glob.glob('output/*.png')
for f in files:
    os.remove(f)

from sort import *

tracker = Sort()

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3),
                           dtype="uint8")

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture("data/test3.mp4")
writer = None
(W, H) = (None, None)

frameIndex = 0

# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]


    dets = []
    dets = detect(frame)
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dets = np.asarray(dets)
    tracks = tracker.update(dets)


    if len(tracks) > 0:
        for box in tracks:
            # extract the bounding box coordinates
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2]), int(box[3]))

            color = [int(c) for c in COLORS[int(box[4]) % len(COLORS)]]
            cv2.rectangle(frame, (x, y), (w, h), color, 2)

            text = "{}".format(int(box[4]))
            text = name[int(box[5])]+"-"+str(int(box[4]))
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # check if the video writer is None
    if writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("out.mp4", fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)


    # write the output frame to disk
    writer.write(frame)

    # increase frame index
    frameIndex += 1

    if frameIndex >= 4000:
        print("[INFO] cleaning up...")
        writer.release()
        vs.release()
        exit()

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()

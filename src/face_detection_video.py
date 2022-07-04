from os import path

import cv2
from tensorflow.python.keras.models import load_model

# load our serialized face detector model from disk
# prototxtPath = r"face_detector\deploy.prototxt"
# weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
# network = cv2.dnn.readNet(prototxtPath, weightsPath)

# # load the face mask detector model from disk
# model = load_model("mask_detector.model")

# # initialize the video stream
# print("[INFO] starting video stream...")
# # vs = VideoStream(src=0).start()
# camera = cv2.VideoCapture(0)

# # loop over the frames from the video stream
# while True:
#     # grab the frame from the video stream and resize it to 400 pixels
#     successs, frame = camera.read()
#     if not successs:
#         break
#     else:
#         frame = cv2.resize(frame, (400, 400))


# def detect_and_predict_mask(frame, network, model):
#     # grab the dimensions of the frame and then construct a blob from it
#     (h, w) = frame.shape[:2]
#     blob = cv2.dnn.blobFromImage(
#         frame,
#         scalefactor=1.0,
#         size=(224, 224),
#         mean=(104.0, 177.0, 123.0),
#     )

#     # pass the blob through the network and obtain the face detections
#     network.setInput(blob)
#     detection = network.forward()
#     print(detection.shape)

#     # initialize list of faces, their corresponding locations,
#     # & the list of predictions from our face mask network
#     faces, locs, preds = [], [], []

#     # loop over the detections
#     # for i in range(0, detection.shape[2]):


prototxtPath = str(path.join("model", "deploy.prototxt"))
weightsPath = str(
    path.join("model", "res10_300x300_ssd_iter_140000.caffemodel")
)
network = cv2.dnn.readNet(prototxtPath, weightsPath)
print(network)
print(cv2.__version__)

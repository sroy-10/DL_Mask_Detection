from os import path
from os.path import dirname, join

import cv2
import numpy as np
from flask import Flask, Response, render_template
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import load_model
from keras.utils import img_to_array

app = Flask(__name__)
camera = cv2.VideoCapture(0)


def generate_frames():
    # load serialized face detector model from disk
    prototxtPath = join(dirname(__file__), "deploy.prototxt")
    weightsPath = join(dirname(__file__), "res10_300x300.caffemodel")
    network = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    model = load_model(join(dirname(__file__), "mask_detector.model"))

    while True:
        # read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:

            # (locs, preds) = detect_and_predict_mask(
            #     frame, network, model
            # )

            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(
                frame,
                scalefactor=1.0,
                size=(224, 224),
                mean=(104.0, 177.0, 123.0),
            )
            network.setInput(blob)
            detection = network.forward()
            print(detection.shape)

            # initialize our list of faces, their corresponding locations, and the list of predictions from our face mask network
            faces, locs, preds = [], [], []

            # loop over the detection
            for i in range(0, detection.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the detection
                confidence = detection[0, 0, i, 2]

                # filter out weak detection by ensuring the confidence is
                # greater than the minimum confidence
                if confidence > 0.5:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the object
                    box = detection[0, 0, i, 3:7] * np.array(
                        [w, h, w, h]
                    )
                    (startX, startY, endX, endY) = box.astype("int")

                    # ensure the bounding boxes fall within the dimensions of
                    # the frame
                    (startX, startY) = (max(0, startX), max(0, startY))
                    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                    # extract the face ROI, convert it from BGR to RGB channel
                    # ordering, resize it to 224x224, and preprocess it
                    face = frame[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face, dtype="float32")
                    face = preprocess_input(face)

                    # add the face and bounding boxes to their respective
                    # lists
                    faces.append(face)
                    locs.append((startX, startY, endX, endY))

                # only make a predictions if at least one face was detected
                if len(faces) > 0:
                    # for faster inference we'll make batch predictions on *all*
                    # faces at the same time rather than one-by-one predictions
                    # in the above `for` loop
                    faces = np.array(faces, dtype="float32")
                    preds = model.predict(faces, batch_size=32)

            # -----------------------------------------------------------

            # loop over the detected face locations and their corresponding locations
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(
                    label, max(mask, withoutMask) * 100
                )

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(
                    frame,
                    label,
                    (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    2,
                )
                cv2.rectangle(
                    frame, (startX, startY), (endX, endY), color, 2
                )

            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


if __name__ == "__main__":
    app.run(debug=False)

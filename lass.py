import time
from imutils.video import VideoStream
import imutils
import wikipedia
import numpy as np
import cv2


def search(text):
    answer = ""
    try:
        try:
            answer = wikipedia.summary(text, sentences=1).replace("'", "").replace("(", "").replace(
                ")", "")

        except Exception:
            print("Error 404")
    except ConnectionError:
        print("Error connecting to Internet")

    return answer


# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "airplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "table",
           "dog", "horse", "motorbike", "person", "potted plant", "sheep",
           "couch", "train", "tv"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] Loading model...")
prototxt = "real_time_object_detection_db/MobileNetSSD_deploy.prototxt.txt"
model = "real_time_object_detection_db/MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

count = 0

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=900)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):

        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > .35:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the prediction on the frame
            label = f"{CLASSES[idx]}: {round(confidence * 100, 2)}%"
            # describe object
            text = f"{search(CLASSES[idx])}"

            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            # code for making the returned definition "pretty" on screen
            split_text = text.split()
            new_line = ""
            max_word_length = 3
            for i in range(len(split_text)):
                new_line += split_text.pop(0) + " "
                if len(new_line.split()) == max_word_length:
                    cv2.putText(frame, new_line, (startX, y + 30 + (15 * i)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200))
                    new_line = ""

                if len(split_text) < max_word_length:
                    cv2.putText(frame, " ".join(split_text[:]), (startX, y + 30 + (15 * i)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 0), 2)
                    break

    # show the output frame
    frame = imutils.resize(frame, width=900)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

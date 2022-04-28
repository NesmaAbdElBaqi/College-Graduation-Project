import Model
import UtilsV2
from tensorflow.keras.models import load_model
import cv2
import os
import numpy as np


def classify_from_video(video_file_path, model, expression_label_encoder):
    # find path of xml file containing haarcascade file
    cascade_path_face = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    # load the harcaascade in the cascade classifier
    face_cascade = cv2.CascadeClassifier(cascade_path_face)
    # load the known faces and embeddings saved in last file

    video_capture = cv2.VideoCapture(video_file_path)
    # loop over frames from the video file stream
    while True:
        # grab the frame from the threaded video stream
        info = []
        ret, frame = video_capture.read()

        if ret != True:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60),
                                              flags=cv2.CASCADE_SCALE_IMAGE)
        # loop over the recognized faces
        for x, y, w, h in faces:
            # rescale the face coordinates
            # draw the predicted face name on the image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

            face_frame = np.copy(frame[x:x + w, y:y + h])
            face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            image = np.array([UtilsV2.preprocess_image(face_frame, img_w, img_h)])
            age_pred, expression_pred = model.predict(image)
            age = age_pred[:, -1][-1]
            expression_index = expression_pred.argmax(-1)[-1]
            label_accuracy_percent = np.round(expression_pred.max(-1)[0] * 100, 2)
            label = expression_label_encoder.classes_[expression_index]
            age = int(np.ceil(age))

            # rescale the face coordinates
            # draw the predicted face name on the image
            # update the list of info
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            info = "Class: {0} accuracy = {1} age = {2}".format(label, label_accuracy_percent, age)
            cv2.putText(frame, info, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            print(info)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img_w = 128
    img_h = 128
    n_classes = 7
    model_path = "best_test2_model.h5"
    model = load_model(model_path)
    load_label_expression_path = "expression_Label_encoder.npy"
    expression_label_encoder = Model.load_label_expression(load_label_expression_path)

    video_file_path = "dataset\OrginalDatabase\DataBase\S1_happy_2.mp4"
    classify_from_video(video_file_path, model, expression_label_encoder)

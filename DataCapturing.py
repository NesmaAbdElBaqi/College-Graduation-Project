import cv2
import glob
import os
global cascade_path_face
global face_cascade


def capture_frame_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)
    frame_faces = []
    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        frame_faces.append(face)
    return frame_faces


def convert_video_labeled_images(video_source_path, images_directory, file_index, image_extension):
    cap = cv2.VideoCapture(video_source_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = round(frame_count / fps)
    rate_capture = int(frame_count / duration)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % rate_capture == 0:
            frame_face = capture_frame_face(frame)
            for face in frame_face:
                cv2.imwrite(images_directory+"_"+str(i+file_index)+image_extension, face)
        i += 1

    cap.release()
    cv2.destroyAllWindows()


def read_all_files_directory(directory, extension):
    return glob.glob(directory+"/*"+extension)


def create_label_directory(label_directory):
    if not os.path.exists(label_directory):
        os.makedirs(label_directory)


def get_video_file_directory(file_path, destination_directory, video_extension):
    file_name = str(file_path).split("\\")[-1].replace(video_extension, "")
    file_index = file_name.split("_")[0]
    file_label = file_name.split("_")[1]
    image_file_directory = destination_directory + "\\" + file_label
    create_label_directory(image_file_directory)
    image_file_path = image_file_directory + "\\" + file_index
    return image_file_directory, image_file_path


def get_video_labeled_frames(source_directory, destination_directory, vidoe_extension=".mp4", image_extension=".jpg"):
    list_files_path = read_all_files_directory(source_directory, vidoe_extension)
    for file_path in list_files_path:
        image_file_directory, image_file_path = get_video_file_directory(file_path, destination_directory, vidoe_extension)
        file_index = len(read_all_files_directory(image_file_directory, image_extension))
        convert_video_labeled_images(file_path, image_file_path, file_index, image_extension)


if __name__ == "__main__":
    source_directory = "DataSet\OriginalDataSet"
    destination_directory = "Dataset\labelFrames"
    vidoe_extension = ".mp4"
    image_extension = ".jpg"

    cascade_path_face = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path_face)

    get_video_labeled_frames(source_directory, destination_directory, vidoe_extension, image_extension)

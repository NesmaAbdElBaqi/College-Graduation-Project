import Model
import Utils
import matplotlib.pyplot as plt

img_w = 128
img_h = 128
n_classes = 7
model = Model.create_model(img_w, img_h, n_classes)
load_label_expression_path = "expression_label_binarizer.npy"
expression_label_binarizer = Model.load_label_expression(load_label_expression_path)


rgb_frame = plt.imread("Dataset\labelFrames\disgust\S1_14.jpg")
gray_frame = Utils.convert_to_gray_scale([rgb_frame])
resized_frame = Utils.resize_image(gray_frame, img_w, img_h)
prediction_class, prediction_reg = model.predict(resized_frame)
label_accuracy_percent = prediction_class.max() * 100
label = expression_label_binarizer.classes_[prediction_class.argmax()]
age = prediction_reg[0][0]
print("Class: {0} \naccuracy = {1} \nage = {2}".format(label, label_accuracy_percent, age))
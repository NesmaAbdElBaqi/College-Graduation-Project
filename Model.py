import numpy as np
import pickle
from sklearn.preprocessing import LabelBinarizer


def load_label_expression(path):
    expression_label_encoder = LabelBinarizer()
    expression_label_encoder.classes_ = np.load(path, allow_pickle=True)
    return expression_label_encoder


import os
from matplotlib import image
import pandas as pd


def read_dataset_x_y1_y2(images_directory_path):
    dataset = []
    for path, sub_dirs, files in os.walk(images_directory_path):
        for name in files:
            file_path = os.path.join(path, name)
            x = image.imread(file_path)
            y1 = path.split('\\')[-1]
            y2 = name.split('_')[0]
            dataset.append([x, y1, y2])
    return dataset


def read_age_dataset_y2(dataset, xlsx_age_dataset, columns, label_column, y2_column):
    age_dataframe = pd.read_excel(xlsx_age_dataset)
    age_dataframe = age_dataframe[columns]
    age_dataframe.dropna(inplace=True)
    for i in range(len(dataset)):
        dataset[i][-1] = age_dataframe.loc[age_dataframe[label_column] == dataset[i][-1]][y2_column].values[0]
    return dataset


if __name__ == "__main__":
    images_directory_path = "Dataset\labeleFrames"
    xlsx_age_dataset = "Dataset\OriginalDataset\summaryParticipansts.xlsx"
    dataset = read_dataset_x_y1_y2(images_directory_path)
    columns = ["Sr. No", "Age"]
    label_column = "Sr. No"
    y2_column = "Age"
    read_age_dataset_y2(dataset, xlsx_age_dataset, columns, label_column, y2_column)

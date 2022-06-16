import numpy as np
import h5py


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    # train_dataset.visit(print)
    # train_dataset.visititems(print)
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features 保存的是训练集里面的图像数据
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels 保存的是训练集里图像对应的分类值

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features 保存的是测试集里面的图像数据
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels 保存的是测试集里图像对应的分类器

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    # print(train_set_y_orig)
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


if __name__ == "__main__":
    load_dataset()

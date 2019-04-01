import numpy as np
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score, cohen_kappa_score
import cv2 as cv
import csv


def one_hot_it(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes

    # Arguments
        label: The 2D array segmentation image label
        label_values

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    x = np.argmax(image, axis=-1)
    return x


def compute_mean_iou(pred, label):
    unique_labels = np.unique(label)
    num_unique_labels = len(unique_labels);

    I = np.zeros(num_unique_labels)
    U = np.zeros(num_unique_labels)

    for index, val in enumerate(unique_labels):
        pred_i = pred == val
        label_i = label == val

        I[index] = float(np.sum(np.logical_and(label_i, pred_i)))
        U[index] = float(np.sum(np.logical_or(label_i, pred_i)))

    mean_iou = np.mean(I / U)
    return mean_iou


# Compute the average segmentation accuracy across all classes
def compute_global_accuracy(pred, label):
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)


# Compute the class-specific segmentation accuracy
def compute_class_accuracies(pred, label, num_classes):
    total = []
    for val in range(num_classes):
        total.append((label == val).sum())

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0

    # If there are no pixels from a certain class in the GT,
    # it returns NAN because of divide by zero
    # Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])

    return accuracies


def evaluate_segmentation(pred, label, num_classes, score_averaging="binary"):
    flat_pred = pred.flatten()
    flat_label = label.flatten()

    global_accuracy = compute_global_accuracy(flat_pred, flat_label)
    class_accuracies = compute_class_accuracies(flat_pred, flat_label, num_classes)

    # prec = precision_score(flat_pred, flat_label, average=score_averaging)
    # rec = recall_score(flat_pred, flat_label, average=score_averaging)
    # f1 = f1_score(flat_pred, flat_label, average=score_averaging)
    prec = precision_score(flat_label, flat_pred, average=score_averaging)
    rec = recall_score(flat_label, flat_pred, average=score_averaging)
    f1 = f1_score(flat_label, flat_pred, average=score_averaging)

    iou = compute_mean_iou(flat_pred, flat_label)
    kp=cohen_kappa_score(flat_label,flat_pred)
    return global_accuracy, class_accuracies, prec, rec, f1, iou,kp


def print_result(pred, label):
    label_values = [[0, 0, 0], [255, 255, 255]]
    num_classes = len(label_values)
    gt = reverse_one_hot(one_hot_it(label, label_values))
    # out = cv.cvtColor(pred, cv.COLOR_GRAY2BGR)
    out = pred
    output_image = reverse_one_hot(one_hot_it(out, label_values))
    accuracy, class_accuracies, prec, rec, f1, iou = evaluate_segmentation(pred=output_image, label=gt,
                                                                           num_classes=num_classes)

    rlt = [prec, rec, iou, f1]
    print("precision: %f" % prec)
    print("recall: %f" % rec)
    print("mIoU: %f " % iou)
    print("\033[0;31mF1: %f \033[0m" % f1)
    return rlt


def csv_out(img_str, short_limit, rlt):
    rlt_csv = []
    for i, j in enumerate(rlt):
        rlt_csv.append("%.5f" % j)

    out_str = [img_str, short_limit]
    out_str.extend(rlt_csv)
    with open("finalv12.csv", 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(out_str)


def in_pic(point, width, height):
    if 0 <= point[0] < width:
        pass
    elif point[0] < 0:
        point[0] = 0
        # print("point wrong width")
    else:
        point[0] = width - 1
        # print("point wrong width")

    if 0 <= point[1] < height:
        pass
    elif point[1] < 0:
        point[1] = 0
        # print("point wrong height")
    else:
        point[1] = height - 1
        # print("point wrong height")
    return point
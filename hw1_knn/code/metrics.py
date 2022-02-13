import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    """
    YOUR CODE IS HERE
    """
    assert y_pred.size == y_true.size, "y_pred and y_true are different size!"
    assert y_pred.size !=0, "label arrays are empty!"
    accuracy = sum(y_pred == y_true) / (y_pred.size)

    TP = sum(y_pred[y_pred == y_true] == "1")

    if sum(y_pred == "1") != 0:
        precision = TP / sum(y_pred == "1")
    else:
        print("Zero instances of class predicted so Precision is equal to 0")
        precision =  0

    if sum(y_true == "1") != 0:
        recall = TP / sum(y_true == "1")
    else: 
        print("Zero instances of class in true samples so Recall is equal to 0")
        recall =  0

    if precision != 0 or recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        print("Both Precision and Recall are zeros, F1 set to 0")
        f1 = 0

    return (precision, recall, f1, accuracy)


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Aassert y_pred.size == y_true.size, "y_pred and y_true are different size!"
    assert y_pred.size != 0, "label arrays are empty!"rguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    """
    YOUR CODE IS HERE
    """
    assert y_pred.size == y_true.size, "y_pred and y_true are different size!"
    assert y_pred.size != 0, "label arrays are empty!"

    return sum(y_pred == y_true) / (y_pred.size)


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """


    """
    YOUR CODE IS HERE
    """
    assert y_pred.size == y_true.size, "y_pred and y_true are different size!"
    assert y_pred.size != 0, "label arrays are empty!"
   
    y_true_mean = np.mean(y_true)
    sd_square_train = np.sum(np.square(y_true - y_true_mean))
    sd_square_predict = np.sum(np.square(y_true - y_pred))

    assert sd_square_train !=0, "Total variance of true samples is zero!"

    return 1 - sd_square_predict / sd_square_train



def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    """
    YOUR CODE IS HERE
    """
    assert y_pred.size == y_true.size, "y_pred and y_true are different size!"
    assert y_pred.size != 0, "label arrays are empty!"

    return np.sum(np.square(y_pred - y_true)) / y_pred.size


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    """
    YOUR CODE IS HERE
    """
    assert y_pred.size == y_true.size, "y_pred and y_true are different size!"
    assert y_pred.size != 0, "label arrays are empty!"

    return np.sum(np.abs(y_pred - y_true)) / y_pred.size

    

import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    softmax_values = softmax(X[i].dot(W))
    correct_probability = softmax_values[y[i]]
    loss += -np.log(correct_probability)
    for j in xrange(num_classes):
        if j == y[i]:
            dW[:, j] -= X[i] * (1 - softmax_values[j])
        else:
            dW[:, j] += X[i] * softmax_values[j]

  loss /= num_train
  dW /= num_train

  loss += reg * np.sum(W * W)
  dW += reg * 2 * W

  return loss, dW


def softmax(scores):
    original_shape = scores.shape
    if len(scores.shape) == 1:
        scores = scores.reshape(1, scores.shape[0])
    scores = scores - scores.max(axis=1).reshape(scores.shape[0], 1)
    scores = np.exp(scores)
    scores = scores / scores.sum(axis=1).reshape(scores.shape[0], 1)
    return scores.reshape(original_shape)


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  num_train = X.shape[0]
  softmax_values = softmax(X.dot(W))
  rows = np.arange(softmax_values.shape[0])
  correct_probability = softmax_values[rows, y]
  loss = np.sum(-np.log(correct_probability))
  softmax_values[rows, y] -= 1
  dW = X.T.dot(softmax_values)

  loss /= num_train
  dW /= num_train

  loss += reg * np.sum(W * W)
  dW += reg * 2 * W

  return loss, dW


def softmax_with_error_signal(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    num_train = X.shape[0]
    softmax_values = softmax(X.dot(W))
    rows = np.arange(softmax_values.shape[0])
    correct_probability = softmax_values[rows, y]
    loss = np.sum(-np.log(correct_probability)) / num_train
    softmax_values[rows, y] -= 1
    dW = X.T.dot(softmax_values)

    dW /= num_train
    error_signal = softmax_values.dot(W.T)

    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    return loss, dW, error_signal

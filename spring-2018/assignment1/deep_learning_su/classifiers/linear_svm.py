import numpy as np
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i]
        loss += margin

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * 2 * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_train = X.shape[0]
  scores = np.matmul(X, W)
  rows = np.arange(len(scores))
  correct_class_score = scores[rows, y].reshape(y.shape[0], 1)
  mask = np.ones(scores.shape)
  mask[rows, y] = 0
  margin = scores - correct_class_score + mask
  positive_margins = margin > 0
  loss = margin[positive_margins].sum()
  loss /= num_train
  loss += reg * np.sum(W * W)

  errors_count = positive_margins.sum(axis=1)
  mask_correct_labels = np.zeros(scores.shape)
  mask_correct_labels[rows, y] = errors_count
  dW = np.matmul(X.T, positive_margins) - np.matmul(X.T, mask_correct_labels)

  dW /= num_train
  dW += reg * 2 * W

  return loss, dW

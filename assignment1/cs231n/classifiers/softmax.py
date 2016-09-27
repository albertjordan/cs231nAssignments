import numpy as np
from random import shuffle
import math

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # Get shapes
  num_classes = W.shape[1]
  num_train = X.shape[0]

  for i in range(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)   #from class notes, to make it more stable

    normalized = np.exp(scores)/ np.sum(np.exp(scores))
    contribution = -np.log(normalized[y[i]])
    loss += contribution

    for j in range(num_classes):
      contribution = np.exp(scores[j])/ np.sum(np.exp(scores))
      dW[:,j] += (contribution  - (j == y[i])) * X[i]


  # Compute average
  loss /= num_train
  dW /= num_train

  # Regularization
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg*W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0

  dW = np.zeros_like(W)

  num_train = X.shape[0]


  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
   

  scores = X.dot(W)
  scores -= np.max(scores)    # class notes...  make sure we have stable ops

  scores = np.exp(scores)     # get the exp, then normalize...
  normalized = (scores.T/np.sum(scores, axis=1)).T    ## .T a trick to get broadcast right
  
  correct = normalized[range(num_train),y]
  contribution = np.sum(-np.log(correct))
  loss += contribution
  loss /= num_train

  # now do the gradient...
  # from lecture notes remember that we need to get the derivative of the function and 
  # multiply that by the input to get the dW
  # 

  #p = np.exp(f)/np.sum(np.exp(f), axis=0)
  ind = np.zeros(scores.shape)
  ind[range(num_train),y] = 1
  dW = np.dot(X.T, (normalized-ind))

  dW /= num_train
 


  # Regularization
  loss += 0.5 * reg * np.sum(W * W)
  #dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  #dW = dW.T
  return loss, dW


import numpy as np

from asgn2.layers import *
from asgn2.fast_layers import *
from asgn2.layer_utils import *

def conv_bn_relu_pool_forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  bn, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  s, relu_cache = relu_forward(bn)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, bn_cache, relu_cache, pool_cache)
  return out, cache

def conv_bn_relu_pool_backward(dout, cache):
  conv_cache, bn_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  dbn = relu_backward(ds, relu_cache)
  da, dgamma, dbeta = spatial_batchnorm_backward(dbn, bn_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db, dgamma, dbeta

def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  bn, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(bn)
  cache = (conv_cache, bn_cache, relu_cache)
  return out, cache

def conv_bn_relu_backward(dout, cache):
  conv_cache, bn_cache, relu_cache = cache
  dbn = relu_backward(dout, relu_cache)
  da, dgamma, dbeta = spatial_batchnorm_backward(dbn, bn_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db, dgamma, dbeta

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
  a, fc_cache = affine_forward(x, w, b)
  bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(bn)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache

def affine_bn_relu_backward(dout, cache):
  fc_cache, bn_cache, relu_cache = cache
  dbn = relu_backward(dout, relu_cache)
  da, dgamma, dbeta = batchnorm_backward(dbn, bn_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db, dgamma, dbeta


class MyCustomConvNet(object):

  def __init__(self, input_dim=(3, 64, 64), num_filters_layer1=32, num_filters_layer2=64, num_filters_layer3=64,
               filter_size_layer1=5, filter_size_layer2=5, filter_size_layer3=5,
               hidden_dim=100, num_classes=3, weight_scale=1e-3, reg=0.0,
               dtype=np.float32,use_saved_weights=False):

    # (Conv-BN-RELU) - (Conv-BN-RELU-POOL)x2 - (FC-BN-RELU) - FC - Softmax

    self.params = {}
    self.reg = reg
    self.dtype = dtype

    stride_pool = 2
    pool_height = 2
    pool_width = 2
    
    H = 1 + (input_dim[1] - pool_height)/stride_pool
    W = 1 + (input_dim[2] - pool_width)/stride_pool

    self.bn_params = []
    self.bn_params = [{'mode': 'train'} for i in xrange(4)]

    if use_saved_weights:
      # Conv1 
      self.params['W1'] = np.load("Saved_Params/W1.npy")
      self.params['b1'] = np.load("Saved_Params/b1.npy")

      # Conv2 
      self.params['W2'] = np.load("Saved_Params/W2.npy")
      self.params['b2'] = np.load("Saved_Params/b2.npy")

      # Conv3 
      self.params['W3'] = np.load("Saved_Params/W3.npy")
      self.params['b3'] = np.load("Saved_Params/b3.npy")

      # Affine1 
      self.params['W4'] = np.load("Saved_Params/W4.npy")
      self.params['b4'] = np.load("Saved_Params/b4.npy")

      # Affine2 
      self.params['W5'] = np.load("Saved_Params/W5.npy")
      self.params['b5'] = np.load("Saved_Params/b5.npy")

      # BN1 
      self.params['beta1'] = np.load("Saved_Params/beta1.npy")
      self.params['gamma1'] = np.load("Saved_Params/gamma1.npy")

      # BN2 
      self.params['beta2'] = np.load("Saved_Params/beta2.npy")
      self.params['gamma2'] = np.load("Saved_Params/gamma2.npy")

      # BN3 
      self.params['beta3'] = np.load("Saved_Params/beta3.npy")
      self.params['gamma3'] = np.load("Saved_Params/gamma3.npy")

      # BN4
      self.params['beta4'] = np.load("Saved_Params/beta4.npy")
      self.params['gamma4'] = np.load("Saved_Params/gamma4.npy")

      # Running means and variances
      self.bn_params[0]['running_mean'] = np.load("Saved_Params/running_mean_0.npy")
      self.bn_params[1]['running_mean'] = np.load("Saved_Params/running_mean_1.npy")
      self.bn_params[2]['running_mean'] = np.load("Saved_Params/running_mean_2.npy")
      self.bn_params[3]['running_mean'] = np.load("Saved_Params/running_mean_3.npy")

      self.bn_params[0]['running_var'] = np.load("Saved_Params/running_var_0.npy")
      self.bn_params[1]['running_var'] = np.load("Saved_Params/running_var_1.npy")
      self.bn_params[2]['running_var'] = np.load("Saved_Params/running_var_2.npy")
      self.bn_params[3]['running_var'] = np.load("Saved_Params/running_var_3.npy")

    else:
      # Conv1 
      self.params['W1'] = weight_scale * np.random.randn(num_filters_layer1, input_dim[0], filter_size_layer1, filter_size_layer1)
      self.params['b1'] = np.zeros(num_filters_layer1)

      # Conv2 
      self.params['W2'] = weight_scale * np.random.randn(num_filters_layer2, num_filters_layer1, filter_size_layer2, filter_size_layer2)
      self.params['b2'] = np.zeros(num_filters_layer2)

      # Conv3 
      self.params['W3'] = weight_scale * np.random.randn(num_filters_layer3, num_filters_layer2, filter_size_layer3, filter_size_layer3)
      self.params['b3'] = np.zeros(num_filters_layer3)

      # Affine1 
      self.params['W4'] = weight_scale * np.random.randn(num_filters_layer3*(H/2)*(W/2), hidden_dim)
      self.params['b4'] = np.zeros(hidden_dim)

      # Affine2 
      self.params['W5'] = weight_scale * np.random.randn(hidden_dim, num_classes)
      self.params['b5'] = np.zeros(num_classes)

      # BN1 
      self.params['beta1'] = np.zeros(num_filters_layer1)
      self.params['gamma1'] = np.ones(num_filters_layer1)

      # BN2 
      self.params['beta2'] = np.zeros(num_filters_layer2)
      self.params['gamma2'] = np.ones(num_filters_layer2)

      # BN3 
      self.params['beta3'] = np.zeros(num_filters_layer3)
      self.params['gamma3'] = np.ones(num_filters_layer3)

      # BN4
      self.params['beta4'] = np.zeros(hidden_dim)
      self.params['gamma4'] = np.ones(hidden_dim)



    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']

    gamma1, beta1 = self.params['gamma1'], self.params['beta1']
    gamma2, beta2 = self.params['gamma2'], self.params['beta2']
    gamma3, beta3 = self.params['gamma3'], self.params['beta3']
    gamma4, beta4 = self.params['gamma4'], self.params['beta4']

    filter_size1 = W1.shape[2]
    conv_param1 = {'stride': 1, 'pad': (filter_size1 - 1) / 2}

    filter_size2 = W2.shape[2]
    conv_param2 = {'stride': 1, 'pad': (filter_size2 - 1) / 2}


    filter_size3 = W3.shape[2]
    conv_param3 = {'stride': 1, 'pad': (filter_size3 - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    mode = 'test' if y is None else 'train'

    for bn_param in self.bn_params:
      bn_param['mode'] = mode

    scores = None

    out1, cache1 = conv_bn_relu_forward(X, W1, b1, gamma1, beta1, conv_param1, self.bn_params[0])
    out2, cache2 = conv_bn_relu_pool_forward(out1, W2, b2, gamma2, beta2, conv_param2, self.bn_params[1],pool_param)
    out3, cache3 = conv_bn_relu_pool_forward(out2, W3, b3, gamma3, beta3, conv_param3, self.bn_params[2],pool_param)
    out4, cache4 = affine_bn_relu_forward(out3, W4, b4, gamma4, beta4, self.bn_params[3])
    scores, cache5 = affine_forward(out4, W5, b5)

    if y is None:
      return scores

    loss, grads = 0, {}

    loss, dscores = softmax_loss(scores, y)

    reg_loss_1 = 0.5 * self.reg * np.sum(self.params['W1'] * self.params['W1'])
    reg_loss_2 = 0.5 * self.reg * np.sum(self.params['W2'] * self.params['W2'])
    reg_loss_3 = 0.5 * self.reg * np.sum(self.params['W3'] * self.params['W3'])
    reg_loss_4 = 0.5 * self.reg * np.sum(self.params['W4'] * self.params['W4'])
    reg_loss_5 = 0.5 * self.reg * np.sum(self.params['W5'] * self.params['W5'])

    loss = loss + reg_loss_1 + reg_loss_2 + reg_loss_3 + reg_loss_4 + reg_loss_5

    dout4, dW5, db5 = affine_backward(dscores, cache5)
    dout3, dW4, db4, dgamma4, dbeta4 = affine_bn_relu_backward(dout4, cache4)
    dout2, dW3, db3, dgamma3, dbeta3 = conv_bn_relu_pool_backward(dout3, cache3)
    dout1, dW2, db2, dgamma2, dbeta2 = conv_bn_relu_pool_backward(dout2, cache2)
    dX, dW1, db1, dgamma1, dbeta1 = conv_bn_relu_backward(dout1, cache1)

    dW5 = dW5 + (self.reg * self.params['W5'])
    dW4 = dW4 + (self.reg * self.params['W4'])
    dW3 = dW3 + (self.reg * self.params['W3'])
    dW2 = dW2 + (self.reg * self.params['W2'])
    dW1 = dW1 + (self.reg * self.params['W1'])

    grads['W1'] = dW1
    grads['b1'] = db1
    grads['W2'] = dW2
    grads['b2'] = db2
    grads['W3'] = dW3
    grads['b3'] = db3
    grads['W4'] = dW4
    grads['b4'] = db4
    grads['W5'] = dW5
    grads['b5'] = db5
    grads['gamma1'] = dgamma1
    grads['beta1'] = dbeta1
    grads['gamma2'] = dgamma2
    grads['beta2'] = dbeta2
    grads['gamma3'] = dgamma3
    grads['beta3'] = dbeta3
    grads['gamma4'] = dgamma4
    grads['beta4'] = dbeta4


    return loss, grads


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################

    stride_pool = 2

    pool_height = 2
    pool_width = 2
    
    H = 1 + (input_dim[1] - pool_height)/stride_pool
    W = 1 + (input_dim[2] - pool_width)/stride_pool

    self.params['W1'] = weight_scale * np.random.randn(num_filters, input_dim[0], filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = weight_scale * np.random.randn(num_filters*H*W, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # conv - relu - 2x2 max pool - affine - relu - affine - softmax

    out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    out2, cache2 = affine_relu_forward(out1, W2, b2)
    scores, cache3 = affine_forward(out2, W3, b3)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)

    reg_loss_1 = 0.5 * self.reg * np.sum(self.params['W1'] * self.params['W1'])
    reg_loss_2 = 0.5 * self.reg * np.sum(self.params['W2'] * self.params['W2'])
    reg_loss_3 = 0.5 * self.reg * np.sum(self.params['W3'] * self.params['W3'])

    loss = loss + reg_loss_1 + reg_loss_2 + reg_loss_3

    dout2, dW3, db3 = affine_backward(dscores, cache3)
    dout1, dW2, db2 = affine_relu_backward(dout2, cache2)
    dX, dW1, db1 = conv_relu_pool_backward(dout1, cache1)

    dW3 = dW3 + (self.reg * self.params['W3'])
    dW2 = dW2 + (self.reg * self.params['W2'])
    dW1 = dW1 + (self.reg * self.params['W1'])

    grads['W1'] = dW1
    grads['b1'] = db1
    grads['W2'] = dW2
    grads['b2'] = db2
    grads['W3'] = dW3
    grads['b3'] = db3

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass

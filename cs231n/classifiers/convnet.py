import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class DeepConvNet(object):
  """
  A multi-layer convolutional network with the following architecture:

  [conv-relu-conv-relu-pool]xN - [affine]xM - [softmax]

  where N = num_conv_layers and M = num_affine_layers

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_conv_layers=1, num_affine_layers=1, num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0, use_batchnorm=False,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_conv_layers: Number of conv-relu-conv-relu-pool layers to use
    - num_affine_layers : Number of affine layers to use (not counting final affine layer)
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - use_batchnorm: use spatial batchnorm after conv layers and vanilla batchnorm after affine layers
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

    self.num_conv_layers = num_conv_layers
    self.num_affine_layers = num_affine_layers

    for i in xrange(num_conv_layers):
      Wi = 'W{0}'.format(2*i + 1)
      bi = 'b{0}'.format(2*i + 1)
      Winext = 'W{0}'.format(2*i + 2)
      binext = 'b{0}'.format(2*i + 2)

      if i == 0:
        conv_shape = (num_filters, input_dim[0], filter_size, filter_size)
        next_conv_shape = (num_filters, num_filters, filter_size, filter_size)
      else:
        conv_shape = (num_filters, num_filters, filter_size, filter_size)
        next_conv_shape = conv_shape

      # Two conv layers
      self.params[Wi] = np.random.normal(0.0, weight_scale, conv_shape)
      self.params[bi] = np.zeros(num_filters,)
      self.params[Winext] = np.random.normal(0.0, weight_scale, next_conv_shape)
      self.params[binext] = np.zeros(num_filters,)


      # TODO: Add spatial batchnorm

    # Find activation map size after all pools
    activation_map_size = num_filters * input_dim[1] / (2**num_conv_layers) * input_dim[2] / (2**num_conv_layers)
    affine_input_dim = activation_map_size

    for i in xrange(num_affine_layers):
      Wi = 'W{0}'.format(2*num_conv_layers + i + 1)
      bi = 'b{0}'.format(2*num_conv_layers + i + 1)
      self.params[Wi] = np.random.normal(0.0, weight_scale, (affine_input_dim, hidden_dim))
      self.params[bi] = np.zeros((hidden_dim,))
      affine_input_dim = hidden_dim

    # Final affine layer
    Wlast = 'W{0}'.format(2*num_conv_layers + num_affine_layers + 1)
    blast = 'b{0}'.format(2*num_conv_layers + num_affine_layers + 1)
    self.params[Wlast] = np.random.normal(0.0, weight_scale, (hidden_dim, num_classes))
    self.params[blast] = np.zeros((num_classes,))

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in sorted(self.params.iteritems()):
      self.params[k] = v.astype(dtype)
      # print "'{0}': {1}".format(k, self.params[k].shape)

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
    hs = []
    caches = []

    hprev = X

    # Perform all convolutions, relus and pools
    for i in xrange(self.num_conv_layers):
      Wi = 'W{0}'.format(2*i + 1)
      bi = 'b{0}'.format(2*i + 1)
      Winext = 'W{0}'.format(2*i + 2)
      binext = 'b{0}'.format(2*i + 2)
      # print self.params[Wi].shape, self.params[Winext].shape
      hi, cachei = conv_relu_forward(hprev, self.params[Wi], self.params[bi], conv_param)
      hinext, cacheinext = conv_relu_pool_forward(hi, self.params[Winext], self.params[binext], conv_param, pool_param)
      hprev = hinext

      hs.append(hi)
      hs.append(hinext)
      caches.append(cachei)
      caches.append(cacheinext)

    # Perform all affine layers
    for i in xrange(self.num_affine_layers):
      Wi = 'W{0}'.format(2*self.num_conv_layers + i + 1)
      bi = 'b{0}'.format(2*self.num_conv_layers + i + 1)

      hi, cachei = affine_relu_forward(hprev, self.params[Wi], self.params[bi])
      hs.append(hi)
      caches.append(cachei)
      hprev = hi


    # Final layer - no relu
    Wlast = 'W{0}'.format(2*self.num_conv_layers + self.num_affine_layers + 1)
    blast = 'b{0}'.format(2*self.num_conv_layers + self.num_affine_layers + 1)
    hlast, cachelast = affine_forward(hprev, self.params[Wlast], self.params[blast])

    hs.append(hlast)
    caches.append(cachelast)

    scores = hlast

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
    loss, dloss = softmax_loss(scores,y)

    dhprev = dloss

    # Gradient for last affine layer
    dhprev, grads[Wlast], grads[blast] = affine_backward(dhprev, caches[2*self.num_conv_layers + self.num_affine_layers])

    # Gradient for affine layers
    for i in reversed(xrange(self.num_affine_layers)):
      Wi = 'W{0}'.format(2*self.num_conv_layers + i + 1)
      bi = 'b{0}'.format(2*self.num_conv_layers + i + 1)

      dhprev, grads[Wi], grads[bi] = affine_relu_backward(dhprev, caches[2*self.num_conv_layers + i])


    # Gradient for conv layers
    for i in reversed(xrange(self.num_conv_layers)):
      Wi = 'W{0}'.format(2*i + 1)
      bi = 'b{0}'.format(2*i + 1)
      Winext = 'W{0}'.format(2*i + 2)
      binext = 'b{0}'.format(2*i + 2)

      dhprev, grads[Winext], grads[binext] = conv_relu_pool_backward(dhprev, caches[2*i + 1])
      dhprev, grads[Wi], grads[bi] = conv_relu_backward(dhprev, caches[2*i])


    # Add L2 regularization
    Ws = ['W{0}'.format(i+1) for i in xrange(2*self.num_conv_layers + self.num_affine_layers)]
    for w in Ws:
      loss += 0.5 * self.reg * (np.sum(np.square(self.params[w])))
      grads[w] += self.params[w] * self.reg

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass

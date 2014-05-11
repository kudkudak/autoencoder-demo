"""
File contains last layer that does the classification
"""

import theano
import sys
import theano.tensor as T
import numpy
import time


class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (e.g., one minibatch of input images)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoint lies

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the target lies
        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                            dtype=theano.config.floatX), name='W' )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                            dtype=theano.config.floatX), name='b' )

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred=T.argmax(self.p_y_given_x, axis=1)

        self.params=[self.W, self.b]
       
        self.output = T.nnet.softmax(T.dot(input, self.W) + self.b) 
 
        m = T.matrix('m') # the data is presented as rasterized images

        self.forward_fnc = theano.function([m], 
            self.output,
            givens = {input: m}
        )



    def forward(self, x):
        return self.forward_fnc(x)

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

          \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
          \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
              \ell (\theta=\{W,b\}, \mathcal{D})


        :param y: corresponds to a vector that gives for each example the
                  correct label;

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # Trick with arange to extract correct labels probability
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


    def dump(self):
        d = {'weights':[self.W.get_value().copy()], 'biases':[self.b.get_value().copy()]}
        return d 

    def load(self, dump):
        self.W.set_value(dump['weights'][0])
        self.b.set_value(dump['biases'][0])
        print "Loaded"


    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero
        one loss over the size of the minibatch
        """
        return T.mean(T.neq(self.y_pred, y))

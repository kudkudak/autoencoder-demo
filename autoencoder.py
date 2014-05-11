"""
File containing main model. This file is worth reading, sorry for lack of documentation
"""


import theano
import theano.tensor as T
import numpy


class dA(object):
   """Autoencoder"""

   def __init__(self, numpy_rng, theano_rng=None, input=None, n_visible=784, n_hidden=500,
              W=None, bhid=None, bvis=None):
       self.n_visible = n_visible
       self.n_hidden = n_hidden

       if not W:
           initial_W = numpy.asarray(numpy_rng.uniform(
                     low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                     high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                     size=(n_visible, n_hidden)), dtype=theano.config.floatX)
           W = theano.shared(value=initial_W, name='W')

       if not bvis:
           bvis = theano.shared(value = numpy.zeros(n_visible,
                                        dtype=theano.config.floatX), name='bvis')

       if not bhid:
           bhid = theano.shared(value=numpy.zeros(n_hidden,
                                             dtype=theano.config.floatX), name='bhid')

       self.W = W
       self.b = bhid
       self.b_prime = bvis
       self.W_prime = self.W.T
       self.theano_rng = theano_rng
       if input == None:
           self.x = T.dmatrix(name='input')
       else:
           self.x = input

       self.params = [self.W, self.b, self.b_prime]
   
       self.output = self.get_hidden_values(input)
 
       self.rec = self.get_reconstructed_input(self.output)

       x = T.matrix('x') # the data is presented as rasterized images
        

       # Compile theano function that returns all values calculated by Autoencoder
       self.rec_fnc = theano.function([x], 
            (self.rec, self.output),
        givens={self.x:x}

       )

   def reconstruct(self, x):
        return self.rec_fnc(x)



   def get_corrupted_input(self, input, corruption_level):
       """ Corrupts input - denoising autoencoder """
       return  self.theano_rng.binomial(size=input.shape, n=1, p=1 - corruption_level) * input

   def forward(self, input):
       return self.get_reconstructed_input(self.get_hidden_values(input))

   def get_hidden_values(self, input):
       return T.nnet.sigmoid(T.dot(input, self.W) + self.b)


   def get_hidden_values(self, input):
       return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

   def get_reconstructed_input(self, hidden ):
       return  T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)


   def get_cost_updates(self, corruption_level, learning_rate):
       tilde_x = self.get_corrupted_input(self.x, corruption_level)
       y = self.get_hidden_values( tilde_x)
       z = self.get_reconstructed_input(y)
       J = T.mean(-T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1 ))

       gparams = T.grad(J, self.params)
       updates = []
       for param, gparam in zip(self.params, gparams):
           updates.append((param, param - learning_rate * gparam))

       return (J, updates)

   def dump(self):
        d = {'weights':[self.W.get_value().copy()], 'biases':[self.b.get_value().copy(), self.b_prime.get_value().copy()]}
        return d

   def load(self, dump):
        self.W.set_value(dump['weights'][0])
        self.b.set_value(dump['biases'][0])
        self.b_prime.set_value(dump['biases'][1])
        print "Loaded"
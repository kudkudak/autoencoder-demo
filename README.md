# Autoencoder MNIST digits recognition demo**
(prepared for student conference **SeMPowisko 2014**)

Demo presents digit recognition using single layer Autoencoder with simple logistic regression for classification
(using learned features). Demo:
    * display quality of reconstruction
    * displays top learned features (new basis)
    * asks user to draw digits and displays **online** probability distribution

Significant part of code is adopted from deeplearning.net tutorial.

[screen1]

[screen2]

[screen3]

### Installation

Install python2 interpreter >= 2.7.3 and pip, then run

`pip install -r requirements.txt`

Or install manually required packages

### Usage

1. Train the network. Type `python2 train.py -h` for possible parameters.

`python2 train.py`

In case of problems try reconfiguring theano to use float32 (in .theanorc file).
But it shouldn't happen.

2. Run demo.

`python2 demo.py`

3. Run drawing application

`python2 drawer.py`

### Autoencoder

**Autoencoder** is a kind of neural network that is "forced" to optimally encode its input.
For more informations see this great UFDL tutorial: [LINK]


### Configuration

There are few parameters possible to tweak

### References

Demo is heavily based on tutorial from deeplearning.net : http://deeplearning.net/tutorial/dA.html
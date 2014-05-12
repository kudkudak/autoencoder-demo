# Autoencoder MNIST digits recognition demo
(prepared for student conference **SeMPowisko 2014**)

Demo presents digit recognition using single layer Autoencoder with simple logistic regression for classification
(using learned features).


Demo:
  
* display quality of reconstruction
  
  * displays top learned features (new basis)
    
  * asks user to draw digits and displays online probability distribution

Significant part of code is adopted from deeplearning.net tutorial.

![](screenshots/1.png =300x200)


### Installation

Install python2 interpreter >= 2.7.3 and pip, then run

`pip install -r requirements.txt`

Or install manually required packages

### Usage

  * Train the network. Type `python2 train.py -h` for possible parameters.

  `python2 train.py`

  In case of problems try reconfiguring theano to use float32 (in .theanorc file).
  But it shouldn't happen.
  You can also use supplied trained model. Unzip model.zip file to the folder

  * Run demo.

  `python2 demo.py`

  * Run drawing application

`python2 drawer.py`

### Configuration

There are few parameters possible to tweak. See -h help in train.py

### Screenshots

![](screenshots/2.png?raw=true =300x200)

![](screenshots/3.png?raw=true =300x200)

### Refereces

**Autoencoder** is a kind of neural network that is "forced" to optimally encode its input.
For more informations see this great UFDL tutorial: [http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity](http://ufldl.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity).

See also original tutorial from deeplearning.net [http://deeplearning.net/tutorial/dA.html](http://deeplearning.net/tutorial/dA.html)

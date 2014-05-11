from optparse import OptionParser
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import time
import matplotlib.pylab as plt
from logistic_regression import *
from autoencoder import dA
import cPickle
from utils import tile_raster_images, load_data

if __name__ == '__main__':
    parser = OptionParser()

    # Define options
    parser.add_option(
        '-h',
        '--n_hidden',
        dest='n_hidden',
        default=1000,
        type=int,
        help='Hidden neurons'
    )
    parser.add_option(
        '-d',
        '--train_da',
        dest='train_da',
        default=True,
        type=bool,
        help='Train denoising autoencoder'
    )
    parser.add_option(
        '-l',
        '--train_logreg',
        dest='train_logreg',
        default=True,
        type=bool,
        help='Train logistic regression'
    )
    parser.add_option(
        '-t',
        '--training_epochs',
        dest='training_epochs',
        default=50,
        type=int,
        help='How long train'
    )
    parser.add_option(
        '-t',
        '--training_epochs',
        dest='training_epochs',
        default=50,
        type=int,
        help='How long train'
    )
    parser.add_option(
        '-b',
        '--noise_level',
        dest='noise_level',
        default=0.3,
        type=float,
        help='How much of the input should be dropped'
    )
    (options, args) = parser.parse_args()

    ############
    #  CONFIG  #
    ############
    learning_rate=0.1
    batch_size=20
    training_epochs=50
    n_hidden=options.n_hidden
    train_da=options.train_da
    train_logreg=options.train_logreg
    noise_level = 0.32


    #################
    # Load datasets #
    #################
    datasets = load_data()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size


    # Here start ugly training script - not really worth reading
    # Training loops come largely from deeplearning.net
    if train_da:
        print "Starting pretraining phase. Pretraining because we are not training for " \
              "discriminative task, but for generative task"

        index = T.lscalar()  # index to a [mini]batch
        x = T.matrix('x')  # the data is presented as rasterized images

        ######################
        # BUILDING THE MODEL #
        ######################

        rng = np.random.RandomState(123)
        theano_rng = RandomStreams(rng.randint(2 ** 30))

        da = dA(numpy_rng=rng, theano_rng=theano_rng, input=x,
                n_visible=28 * 28, n_hidden=n_hidden)


        cost, updates = da.get_cost_updates(corruption_level=noise_level,
                                    learning_rate=learning_rate)

        # Build traninig function
        train_da = theano.function([index],
            cost, updates=updates,
            givens = {x: train_set_x[index * batch_size: (index + 1) * batch_size]}
        )


        ################
        # PRE TRAINING #
        ################

        start_time = time.clock()
        # go through training epochs
        for epoch in xrange(training_epochs):
            # go through trainng set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(train_da(batch_index))
            print 'Pre-training epoch %d, cost ' % epoch, numpy.mean(c)


        end_time = time.clock()
        pretraining_time = (end_time - start_time)
        print ('Pre-training took %f minutes' % (pretraining_time / 60.))

        image = tile_raster_images(
                X=da.W.get_value(borrow=True).T,
                img_shape=(28, 28), tile_shape=(10, 10),
                tile_spacing=(1, 1))

        plt.imshow(image)
        plt.show()

        print "Saving pre-trained model"

        cPickle.dump(da.dump(), open("demo_da.pkl","w"))
        print "Model saved"

    if train_logreg:
        ##############
        ## TRAINING ##
        ##############

        print "Starting training phase"
        print "Note! We are also modifying W matrix of the autoencoder"

        # Prepare shared variables
        index = T.lscalar() # index to a [mini]batch
        x = T.matrix('x') # the data is presented as rasterized images
        y = T.ivector('y') # the labels are presented as 1D vector of


        # Load Denoising Autoencoder
        rng = np.random.RandomState(123)
        theano_rng = RandomStreams(rng.randint(2 ** 30))
        da = dA(numpy_rng=rng, theano_rng=theano_rng, input=x,
                n_visible=28 * 28, n_hidden=n_hidden)
        import cPickle
        da.load(cPickle.load(open("demo_da.pkl","r")))

        input_logreg = da.get_hidden_values(x)

        image = tile_raster_images(
                X=da.W.get_value(borrow=True).T,
                img_shape=(28, 28), tile_shape=(10, 10),
                tile_spacing=(1, 1))

        plt.imshow(image)
        plt.show()


        # Build training function
        # allocate symbolic variables for the data
                            # [int] labels
        classifier = LogisticRegression(input=input_logreg, n_in=n_hidden, n_out=10)
        cost = classifier.negative_log_likelihood(y)


        test_model = theano.function(inputs=[index],
                outputs=classifier.errors(y),
                givens={
                    x: test_set_x[index * batch_size: (index + 1) * batch_size],
                    y: test_set_y[index * batch_size: (index + 1) * batch_size]})

        validate_model = theano.function(inputs=[index],
                outputs=classifier.errors(y),
                givens={
                    x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                    y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

        # compute the gradient of cost with respect to theta = (W,b)
        g_W = T.grad(cost=cost, wrt=classifier.W)
        g_b = T.grad(cost=cost, wrt=classifier.b)

        g_da_W = T.grad(cost=cost, wrt=da.W)
        g_da_b = T.grad(cost=cost, wrt=da.b)

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs.
        updates = [(classifier.W, classifier.W - learning_rate * g_W),
                    (classifier.b, classifier.b - learning_rate * g_b),
                    (da.W, da.W - learning_rate * g_da_W),
                    (da.b, da.b - learning_rate * g_da_b)]
        # compiling a Theano function `train_model` that returns the cost, but in
        # the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(inputs=[index],
                outputs=cost,
                updates=updates,
                givens={
                    x: train_set_x[index * batch_size:(index + 1) * batch_size],
                    y: train_set_y[index * batch_size:(index + 1) * batch_size]})


        # early-stopping parameters
        patience = 5000 # look as this many examples regardless
        patience_increase = 2 # wait this much longer when a new best is
                                        # found
        improvement_threshold = 0.995 # a relative improvement of this much is
                                        # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                        # go through this many
                                        # minibatche before checking the network
                                        # on the validation set; in this case we
                                        # check every epoch

        best_params = None
        best_validation_loss = numpy.inf
        test_score = 0.
        start_time = time.clock()

        done_looping = False
        epoch = 0
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):

                minibatch_avg_cost = train_model(minibatch_index)
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i)
                                            for i in xrange(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)

                    print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                        (epoch, minibatch_index + 1, n_train_batches,
                        this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss * \
                            improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        # test it on the test set

                        test_losses = [test_model(i)
                                        for i in xrange(n_test_batches)]
                        test_score = numpy.mean(test_losses)

                        print((' epoch %i, minibatch %i/%i, test error of best'
                            ' model %f %%') %
                            (epoch, minibatch_index + 1, n_train_batches,
                                test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = time.clock()
        print(('Optimization complete with best validation score of %f %%,'
                'with test performance %f %%') %
                        (best_validation_loss * 100., test_score * 100.))
        print 'The code run for %d epochs, with %f epochs/sec' % (
            epoch, 1. * epoch / (end_time - start_time))
        print >> sys.stderr, ('The code for file ' +
                                os.path.split(__file__)[1] +
                                ' ran for %.1fs' % ((end_time - start_time)))

        print "Saving trained model"
        import cPickle
        cPickle.dump((da.dump(),classifier.dump()), open("demo_model.pkl","w"))
        print "Model saved"


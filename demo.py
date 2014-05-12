import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import matplotlib.pylab as plt
from logistic_regression import *
from autoencoder import dA
from utils import load_data, tile_raster_images



if __name__ == "__main__":
    parser = OptionParser()

    # Define options
    parser.add_option(
        '-k',
        '--n_hidden',
        dest='n_hidden',
        default=1000,
        type=int,
        help='Hidden neurons'
    )
    (options, args) = parser.parse_args()

    n_hidden = options.n_hidden

    print "Supplied {0} n_hidden. Please remember to pass correct number of hidden neurons (default is 1000)".format(n_hidden)


    # load dataset
    datasets = load_data()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]


    print "Loading learned model"

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
    da_dumped, logreg_dumped = cPickle.load(open("demo_model.pkl","r"))

    print logreg_dumped

    da.load(da_dumped)

    input_logreg = da.get_hidden_values(x)

    # Load logistic regression
    classifier = LogisticRegression(input=input_logreg, n_in=n_hidden, n_out=10)

    classifier.load(logreg_dumped)




    for i in [0,1,3,5]:
        rec,rec_hid = da.reconstruct(train_set_x.get_value(borrow=True)[i:i+1,:])
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(train_set_x.get_value(borrow=True)[i:i+1,:].reshape(28,28), cmap='gray')
        ax1.set_title("Original digit")
        ax2.imshow(rec.reshape(28,28), cmap='gray')
        ax2.set_title("Reconstruction by "+str(28*28)+"x"+n_hidden+" autoencoder network")
        plt.show()


        sort_pairs = [(val, idx) for idx, val in enumerate(rec_hid.reshape(-1))]
        sort_pairs.sort(reverse=True)
        print sort_pairs

        image = tile_raster_images(
                X=da.W.get_value(borrow=True).T[[x[1] for x in sort_pairs],],
                img_shape=(28, 28), tile_shape=(5, 5),
                tile_spacing=(1, 1))
        plt.imshow(image, cmap='gray')
        plt.show()



    ######## Plotting results #############3

    print "Testing on"
    print type(test_set_x)
    print type(test_set_y)
    test_set_x = test_set_x.get_value(borrow=True)
    test_set_y = test_set_y.get_value(borrow=True)

    import skimage
    import skimage.transform
    def im_scale(img, scale_factor):
        zoomed_img = np.zeros_like(img, dtype=img.dtype)
        zoomed = skimage.transform.rescale(img, scale_factor)

        if scale_factor >= 1.0:
            shift_x = (zoomed.shape[0] - img.shape[0]) // 2
            shift_y = (zoomed.shape[1] - img.shape[1]) // 2
            zoomed_img[:,:] = zoomed[shift_x:shift_x+img.shape[0], shift_y:shift_y+img.shape[1]]
        else:
            shift_x = (img.shape[0] - zoomed.shape[0]) // 2
            shift_y = (img.shape[1] - zoomed.shape[1]) // 2
            zoomed_img[shift_x:shift_x+zoomed.shape[0], shift_y:shift_y+zoomed.shape[1]] = zoomed

        return zoomed_img
    import threading
    from drawer import draw_main
    print "=="
    m = T.matrix('m') # the data is presented as rasterized images
    forward_fnc = theano.function([m],
        classifier.output,
        givens = {da.x: m}
    )
    from PIL import Image
    import numpy as np
    import numpy
    import time


    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import timeit

    clock = timeit.default_timer

    fig, ax = plt.subplots()

    alphab = ['0', '1','2','3','4','5','6','7','8','9']
    frequencies = [1.0]*10

    pos = np.arange(len(alphab))
    width = 1.0     # gives histogram aspect to the bar diagram
    ax.set_xticks(pos + (width / 2))
    ax.set_xticklabels(alphab)

    rects = plt.bar(pos, frequencies, width, color='r')
    start = clock()

    def animate(arg, rects):
        frameno, frequencies = arg
        for rect, f in zip(rects, frequencies):
            rect.set_height(f)
        print("FPS: {:.2f}".format(frameno / (clock() - start)))




    def step():
        previous = [0.1]*10
        frame = 0
        while True:
            frame += 1
            try:
                img2 = numpy.asarray(Image.open("digit.eps").resize((28,28)))[:,:,0]
                img = numpy.array(img2)
                for i in xrange(28):
                    for j in xrange(28):
                        img[i,j] = 1.0 if img[i,j] < 120 else 0.0
                previous = forward_fnc(img.reshape(-1).reshape(1,-1))[0]
            except Exception ,e:
                print "Error",e

            yield frame, previous


    ani = animation.FuncAnimation(fig, animate, step, interval=10,
                                  repeat=False, blit=False, fargs=(rects,))


    plt.show()


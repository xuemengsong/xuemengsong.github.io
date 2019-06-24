import os
import sys
import time
import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import gzip
import cPickle
import pickle
from  datetime import *
import time
theano.config.floatX= 'float32'
class dA_joint(object):

    def __init__(
            self,
            numpy_rng,
            theano_rng=None,
            input1_v=None,
            input2_v=None,
            input3_v=None,
            input1_c=None,
            input2_c=None,
            input3_c=None,
            n_visible1_v=4096,
            n_visible2_v=4096,
            n_visible1_c=2327,
            n_visible2_c=2327,
            n_hidden_v=None,
            n_hidden_c=None,
            W1_c=None,
            bhid1_c=None,
            bvis1_c=None,
            W2_c=None,
            bhid2_c=None,
            bvis2_c=None,
            W1_v=None,
            bhid1_v=None,
            bvis1_v=None,
            W2_v=None,
            bhid2_v=None,
            bvis2_v=None,
            lamda=None,
            mu=None,
            beta=None,
            theta=None,
            momentum=0.9
    ):
        self.n_visible1_v = n_visible1_v
        self.n_visible2_v = n_visible2_v
        self.n_hidden_v = n_hidden_v
        self.n_visible1_c = n_visible1_c
        self.n_visible2_c = n_visible2_c
        self.n_hidden_c = n_hidden_c
        self.lamda = lamda
        self.mu = mu
        self.beta = beta
        self.theta = theta
        self.momentum = momentum
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if not W1_v:

            initial_W1_v = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden_v + n_visible1_v)),
                    high=4 * numpy.sqrt(6. / (n_hidden_v + n_visible1_v)),
                    size=(n_visible1_v, n_hidden_v)
                ),
                dtype=theano.config.floatX
            )
            W1_v = theano.shared(value=initial_W1_v, name='W1_v', borrow=True)
        if not W1_c:
            initial_W1_c = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden_c + n_visible1_c)),
                    high=4 * numpy.sqrt(6. / (n_hidden_c + n_visible1_c)),
                    size=(n_visible1_c, n_hidden_c)
                ),
                dtype=theano.config.floatX
            )
            W1_c = theano.shared(value=initial_W1_c, name='W1_c', borrow=True)
        if not W2_v:
            initial_W2_v = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden_v + n_visible2_v)),
                    high=4 * numpy.sqrt(6. / (n_hidden_v + n_visible2_v)),
                    size=(n_visible2_v, n_hidden_v)
                ),
                dtype=theano.config.floatX
            )
            W2_v = theano.shared(value=initial_W2_v, name='W2_v', borrow=True)
        if not W2_c:
            initial_W2_c = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden_c + n_visible2_c)),
                    high=4 * numpy.sqrt(6. / (n_hidden_c + n_visible2_c)),
                    size=(n_visible2_c, n_hidden_c)
                ),
                dtype=theano.config.floatX
            )
            W2_c = theano.shared(value=initial_W2_c, name='W2_c', borrow=True)

        if not bvis1_v:
            bvis1_v = theano.shared(
                value=numpy.zeros(
                    n_visible1_v,
                    dtype=theano.config.floatX
                ),
                name='b1pv',
                borrow=True
            )
        if not bvis2_v:
            bvis2_v = theano.shared(
                value=numpy.zeros(
                    n_visible2_v,
                    dtype=theano.config.floatX
                ),
                name='b2pv',
                borrow=True
            )

        if not bhid1_v:
            bhid1_v = theano.shared(
                value=numpy.zeros(
                    n_hidden_v,
                    dtype=theano.config.floatX
                ),
                name='b1v',
                borrow=True
            )
        if not bhid2_v:
            bhid2_v = theano.shared(
                value=numpy.zeros(
                    n_hidden_v,
                    dtype=theano.config.floatX
                ),
                name='b2v',
                borrow=True
            )
        if not bvis1_c:
            bvis1_c = theano.shared(
                value=numpy.zeros(
                    n_visible1_c,
                    dtype=theano.config.floatX
                ),
                name='b1pc',
                borrow=True
            )
        if not bvis2_c:
            bvis2_c = theano.shared(
                value=numpy.zeros(
                    n_visible2_c,
                    dtype=theano.config.floatX
                ),
                name='b2pc',
                borrow=True
            )

        if not bhid1_c:
            bhid1_c = theano.shared(
                value=numpy.zeros(
                    n_hidden_c,
                    dtype=theano.config.floatX
                ),
                name='b1c',
                borrow=True
            )
        if not bhid2_c:
            bhid2_c = theano.shared(
                value=numpy.zeros(
                    n_hidden_c,
                    dtype=theano.config.floatX
                ),
                name='b2c',
                borrow=True
            )

        self.W1_v = W1_v
        self.W2_v = W2_v
        self.W1_c = W1_c
        self.W2_c = W2_c

        self.b1_v = bhid1_v
        self.b2_v = bhid2_v
        self.b1_c = bhid1_c
        self.b2_c = bhid2_c

        self.b1_prime_v = bvis1_v
        self.b2_prime_v = bvis2_v
        self.b1_prime_c = bvis1_c
        self.b2_prime_c = bvis2_c

        self.W1_prime_v = self.W1_v.T
        self.W2_prime_v = self.W2_v.T
        self.W1_prime_c = self.W1_c.T
        self.W2_prime_c = self.W2_c.T


        self.theano_rng = theano_rng
        self.L2_sqr = (
            (self.W1_v ** 2).mean() + (self.W2_v ** 2).mean() + (self.W1_c ** 2).mean() + (self.W2_c ** 2).mean()
            + (self.b1_v ** 2).mean() + (self.b2_v ** 2).mean() + (self.b1_c ** 2).mean() + (self.b2_c ** 2).mean()
            + (self.b1_prime_v ** 2).mean() + (self.b2_prime_v ** 2).mean() + (self.b1_prime_c ** 2).mean() + (self.b2_prime_c ** 2).mean()
        )
        # if no input is given, generate a variable representing the input
        if input1_v is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x1_v = T.dmatrix(name='input1_v',dtype='float32')
            self.x2_v = T.dmatrix(name='input2_v',dtype='float32')
            self.x3_v = T.dmatrix(name='input3_v',dtype='float32')
            self.x1_c = T.dmatrix(name='input1_c',dtype='float32')
            self.x2_c = T.dmatrix(name='input2_c',dtype='float32')
            self.x3_c = T.dmatrix(name='input3_c',dtype='float32')

        else:
            self.x1_v = input1_v
            self.x2_v = input2_v
            self.x3_v = input3_v
            self.x1_c = input1_c
            self.x2_c = input2_c
            self.x3_c = input3_c

        self.params = [self.W1_v, self.b1_v, self.b1_prime_v,
                       self.W2_v, self.b2_v, self.b2_prime_v,
                       self.W1_c, self.b1_c, self.b1_prime_c,
                       self.W2_c, self.b2_c, self.b2_prime_c
                       ]
        # end-snippet-1
        self.output1_v = T.nnet.hard_sigmoid (T.dot(self.x1_v, self.W1_v) + self.b1_v)
        self.output2_v = T.nnet.hard_sigmoid (T.dot(self.x2_v, self.W2_v) + self.b2_v)
        self.output3_v = T.nnet.hard_sigmoid (T.dot(self.x3_v, self.W2_v) + self.b2_v)
        self.output1_c = T.nnet.hard_sigmoid (T.dot(self.x1_c, self.W1_c) + self.b1_c)
        self.output2_c = T.nnet.hard_sigmoid (T.dot(self.x2_c, self.W2_c) + self.b2_c)
        self.output3_c = T.nnet.hard_sigmoid (T.dot(self.x3_c, self.W2_c) + self.b2_c)

        self.output1t_v = T.transpose(self.output1_v)
        self.output2t_v = T.transpose(self.output2_v)
        self.output3t_v = T.transpose(self.output3_v)
        self.output1t_c = T.transpose(self.output1_c)
        self.output2t_c = T.transpose(self.output2_c)
        self.output3t_c = T.transpose(self.output3_c)

        self.rec1_v = T.nnet.hard_sigmoid (T.dot(self.output1_v, self.W1_prime_v) + self.b1_prime_v)
        self.rec2_v = T.nnet.hard_sigmoid (T.dot(self.output2_v, self.W2_prime_v) + self.b2_prime_v)
        self.rec3_v = T.nnet.hard_sigmoid (T.dot(self.output3_v, self.W2_prime_v) + self.b2_prime_v)
        self.rec1_c = T.nnet.hard_sigmoid (T.dot(self.output1_c, self.W1_prime_c) + self.b1_prime_c)
        self.rec2_c = T.nnet.hard_sigmoid (T.dot(self.output2_c, self.W2_prime_c) + self.b2_prime_c)
        self.rec3_c = T.nnet.hard_sigmoid (T.dot(self.output3_c, self.W2_prime_c) + self.b2_prime_c)


    def get_hidden_values(self, input1_v, input2_v, input3_v,input1_c, input2_c, input3_c):
        """ Computes the values of the hidden layer """
        return T.nnet.hard_sigmoid (T.dot(input1_v, self.W1_v) + self.b1_v), T.nnet.hard_sigmoid (T.dot(input2_v, self.W2_v) + self.b2_v), T.nnet.hard_sigmoid (
            T.dot(input3_v, self.W2_v) + self.b2_v),T.nnet.hard_sigmoid (T.dot(input1_c, self.W1_c) + self.b1_c), T.nnet.hard_sigmoid (T.dot(input2_c, self.W2_c) + self.b2_c), T.nnet.hard_sigmoid (
            T.dot(input3_c, self.W2_c) + self.b2_c)

    def get_reconstructed_input(self, hidden1_v, hidden2_v, hidden3_v,hidden1_c, hidden2_c, hidden3_c):

        a = T.nnet.hard_sigmoid (T.dot(hidden1_v, self.W1_prime_v) + self.b1_prime_v)
        b = T.nnet.hard_sigmoid (T.dot(hidden2_v, self.W2_prime_v) + self.b2_prime_v)
        c = T.nnet.hard_sigmoid (T.dot(hidden3_v, self.W2_prime_v) + self.b2_prime_v)
        d = T.nnet.hard_sigmoid (T.dot(hidden1_c, self.W1_prime_c) + self.b1_prime_c)
        e = T.nnet.hard_sigmoid (T.dot(hidden2_c, self.W2_prime_c) + self.b2_prime_c)
        f = T.nnet.hard_sigmoid (T.dot(hidden3_c, self.W2_prime_c) + self.b2_prime_c)
        return a, b, c, d, e, f

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        y1_v, y2_v, y3_v, y1_c, y2_c, y3_c = self.get_hidden_values(self.x1_v, self.x2_v, self.x3_v,self.x1_c, self.x2_c, self.x3_c)
        y1t_v = T.transpose(y1_v)
        y2t_v = T.transpose(y2_v)
        y3t_v = T.transpose(y3_v)
        y2t_c = T.transpose(y2_c)
        y3t_c = T.transpose(y3_c)
        z1_v, z2_v, z3_v, z1_c, z2_c, z3_c = self.get_reconstructed_input(y1_v, y2_v, y3_v, y1_c, y2_c, y3_c)
        L_x1_v = T.mean((z1_v - self.x1_v) ** 2)
        L_x2_v = T.mean((z2_v - self.x2_v) ** 2)
        L_x3_v = T.mean((z3_v - self.x3_v) ** 2)
        L_x1_c = T.mean((z1_c - self.x1_c) ** 2)
        L_x2_c = T.mean((z2_c - self.x2_c) ** 2)
        L_x3_c = T.mean((z3_c - self.x3_c) ** 2)
        d_x1_x2_v = T.dot(y1_v, y2t_v).diagonal()
        d_x1_x3_v = T.dot(y1_v, y3t_v).diagonal()
        d_x1_x2_c = T.dot(y1_c, y2t_c).diagonal()
        d_x1_x3_c = T.dot(y1_c, y3t_c).diagonal()
        L_mod = T.mean(T.nnet.hard_sigmoid(T.dot(y1t_v, y1_c).diagonal())) + T.mean(T.nnet.hard_sigmoid(T.dot(y2t_v, y2_c).diagonal())) + T.mean(T.nnet.hard_sigmoid(T.dot(y3t_v, y3_c).diagonal()))
        L_sup = T.mean(T.nnet.hard_sigmoid((d_x1_x2_v - d_x1_x3_v) + self.theta * (d_x1_x2_c - d_x1_x3_c)))
        L_sqr = self.L2_sqr
        L_123 = L_x1_v + L_x2_v + L_x3_v + L_x1_c + L_x2_c + L_x3_c
        L_rec = self.mu * L_123 + self.lamda * L_sqr
        cost = L_rec - L_sup - self.beta * L_mod

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []

        for p, g in zip(self.params, gparams):
            mparam_i = theano.shared(numpy.zeros(p.get_value().shape, dtype=theano.config.floatX))
            v = self.momentum * mparam_i - learning_rate * g
            updates.append((mparam_i, v))
            updates.append((p, p + v))

        return (cost, updates, L_rec, L_sup, L_mod, L_sqr, L_123, d_x1_x2_v, d_x1_x2_c, d_x1_x3_v, d_x1_x3_c)


def BPR_DAE(learning_rate=0.01, batch_size=128, epoch_time=30, max_patience=3):
    fb = open('ijk_pretrain_da_1layerVisual_best' + str(learning_rate) + '.txt', 'a+')
    fi = open('ijk_pretrain_da_1layerVisual_cost' + str(learning_rate) + '.txt', 'a+')


    # with open("/home/liujinhuan/MM/bottom_shoes_test/Visual/AUC_new_dataset_train_811_norm.pkl", "r") as f:
    #     train_set =  numpy.asarray(cPickle.load(f),dtype='float32')
    # with open("/home/liujinhuan/MM/bottom_shoes_test/Visual/AUC_new_dataset_valid_811_norm.pkl", "r") as f:
    #     valid_set =  numpy.asarray(cPickle.load(f),dtype='float32')
    # with open("/home/liujinhuan/MM/bottom_shoes_test/Visual/AUC_new_dataset_test_811_norm.pkl", "r") as f:
    #     test_set = numpy.asarray(cPickle.load(f),dtype='float32')
    # with open("/home/liujinhuan/MM/bottom_shoes_test/Textual/AUC_new_dataset_unified_text_train8110.pkl", "rb") as f:
    #     train_txt_set =  numpy.asarray(cPickle.load(f),dtype='float32')
    # with open("/home/liujinhuan/MM/bottom_shoes_test/Textual/AUC_new_dataset_unified_text_valid8110.pkl", "rb") as f:
    #     valid_txt_set =  numpy.asarray(cPickle.load(f),dtype='float32')
    # with open("/home/liujinhuan/MM/bottom_shoes_test/Textual/AUC_new_dataset_unified_text_test8110.pkl", "rb") as f:
    #     test_txt_set = numpy.asarray(cPickle.load(f),dtype='float32')
    with open("AUC_new_dataset_train_811_norm.pkl", "r") as f:
        train_set =  numpy.asarray(cPickle.load(f),dtype='float32')
    print 1
    with open("AUC_new_dataset_valid_811_norm.pkl", "r") as f:
        valid_set =  numpy.asarray(cPickle.load(f),dtype='float32')
    print 2
    with open("AUC_new_dataset_test_811_norm.pkl", "r") as f:
        test_set = numpy.asarray(cPickle.load(f),dtype='float32')
    print 3
    with open("AUC_new_dataset_unified_text_train8110.pkl", "r") as f:
        train_txt_set =  numpy.asarray(cPickle.load(f),dtype='float32')
    print 4
    with open("AUC_new_dataset_unified_text_valid8110.pkl", "r") as f:
        valid_txt_set =  numpy.asarray(cPickle.load(f),dtype='float32')
    print 5
    with open("AUC_new_dataset_unified_text_test8110.pkl", "r") as f:
        test_txt_set = numpy.asarray(cPickle.load(f),dtype='float32')
    print 6


    train_set_size = train_set[0].shape[0]
    valid_set_size = valid_set[0].shape[0]
    test_set_size = test_set[0].shape[0]
    n_train_batches = int(train_set_size / batch_size )

    train_set_xi_v, train_set_xj_v, train_set_xk_v = theano.tensor._shared(train_set[0]), theano.tensor._shared(train_set[1]), theano.tensor._shared(train_set[2])
    valid_set_xi_v, valid_set_xj_v, valid_set_xk_v = theano.tensor._shared(valid_set[0]), theano.tensor._shared(valid_set[1]), theano.tensor._shared(valid_set[2])
    test_set_xi_v, test_set_xj_v, test_set_xk_v = theano.tensor._shared(test_set[0]), theano.tensor._shared(test_set[1]), theano.tensor._shared(test_set[2])
    train_set_xi_c, train_set_xj_c, train_set_xk_c = theano.tensor._shared(train_txt_set[0]), theano.tensor._shared(train_txt_set[1]), theano.tensor._shared(train_txt_set[2])
    valid_set_xi_c, valid_set_xj_c, valid_set_xk_c = theano.tensor._shared(valid_txt_set[0]), theano.tensor._shared(valid_txt_set[1]), theano.tensor._shared(valid_txt_set[2])
    test_set_xi_c, test_set_xj_c, test_set_xk_c = theano.tensor._shared(test_txt_set[0]), theano.tensor._shared(test_txt_set[1]), theano.tensor._shared(test_txt_set[2])

    print 'loaded data'

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    improvement_threshold = 0.995
    best_validation_ndcg_score = 0.0
    test_ndcg_score = 0.0
    iter = [0.1, 0.1, 0.1, 0.1]
    best_valid_iter = iter
    ###############
    # TRAIN MODEL #
    ###############
    print 'iteration start'
    start_time = time.clock()
    count = 0
    for _lamda in range(-2, -1, 1):
        for _mu in range(-2, -1, 1):
            for _beta in range( -3, -2, 1):
                for n_hidden in [512]:
                    index = T.lscalar()
                    xi_v = T.matrix('xi_v')
                    xj_v = T.matrix('xj_v')
                    xk_v = T.matrix('xk_v')
                    xi_c = T.matrix('xi_c')
                    xj_c = T.matrix('xj_c')
                    xk_c = T.matrix('xk_c')
                    count = count + 1
                    lamda = 10 ** (_lamda)
                    mu = 10 ** (_mu)
                    beta = (10 ** (_beta))
                    theta = 1


                    iter = [lamda, mu, beta, n_hidden]
                    print '%i iterations' % (count)
                    da = dA_joint(
                        numpy_rng=rng,
                        theano_rng=theano_rng,
                        input1_v=xi_v,
                        input2_v=xj_v,
                        input3_v=xk_v,
                        input1_c=xi_c,
                        input2_c=xj_c,
                        input3_c=xk_c,
                        n_visible1_v=4096,
                        n_visible2_v=4096,
                        n_visible1_c=2327,
                        n_visible2_c=2327,
                        n_hidden_v=n_hidden,
                        n_hidden_c=n_hidden,
                        lamda=lamda,
                        mu=mu,
                        beta=beta,
                        theta=theta,
                        momentum=0.9
                    )

                    cost, updates, L_rec_cost, L_sup_cost, L_mod_cost, L_sqr, L_123, d_x1_x2_v, d_x1_x2_c, d_x1_x3_v, d_x1_x3_c = da.get_cost_updates(
                        corruption_level=0.3,
                        learning_rate=learning_rate
                    )

                    train_model = theano.function(
                        [index],
                        [cost, L_rec_cost, L_sup_cost, L_mod_cost, L_sqr, L_123, d_x1_x2_v, d_x1_x2_c, d_x1_x3_v, d_x1_x3_c],
                        updates=updates,
                        givens={
                            xi_v: (train_set_xi_v[index * batch_size: (index + 1) * batch_size]),
                            xj_v: (train_set_xj_v[index * batch_size: (index + 1) * batch_size]),
                            xk_v: (train_set_xk_v[index * batch_size: (index + 1) * batch_size]),
                            xi_c: (train_set_xi_c[index * batch_size: (index + 1) * batch_size]),
                            xj_c: (train_set_xj_c[index * batch_size: (index + 1) * batch_size]),
                            xk_c: (train_set_xk_c[index * batch_size: (index + 1) * batch_size])
                        }
                    )
                     
                    valid_xi_v = theano.function(
                        [],
                        outputs=da.output1_v,
                        givens={
                            xi_v: valid_set_xi_v
                        },
                        allow_input_downcast=True,
                        name='valid_xi_v'
                    )
                    valid_xj_v = theano.function(
                        [],
                        outputs=da.output2t_v,
                        givens={
                            xj_v: valid_set_xj_v
                        },
                        allow_input_downcast=True,
                        name='valid_xj_v'
                    )
                    valid_xk_v = theano.function(
                        [],
                        outputs=da.output3t_v,
                        givens={
                            xk_v: valid_set_xk_v
                        },
                        allow_input_downcast=True,
                        name='valid_xk_v'
                    )
                    valid_xi_c = theano.function(
                        [],
                        outputs=da.output1_c,
                        givens={
                            xi_c: valid_set_xi_c
                        },
                        allow_input_downcast=True,
                        name='valid_xi_c'
                    )
                    valid_xj_c = theano.function(
                        [],
                        outputs=da.output2t_c,
                        givens={
                            xj_c: valid_set_xj_c
                        },
                        allow_input_downcast=True,
                        name='valid_xj_c'
                    )
                    valid_xk_c = theano.function(
                        [],
                        outputs=da.output3t_c,
                        givens={
                            xk_c: valid_set_xk_c
                        },
                        allow_input_downcast=True,
                        name='valid_xk_c'
                    )
                    test_xi_v = theano.function(
                        [],
                        outputs=da.output1_v,
                        givens={
                            xi_v: test_set_xi_v
                        },
                        allow_input_downcast=True,
                        name='test_xi_v'
                    )
                    test_xj_v = theano.function(
                        [],
                        outputs=da.output2t_v,
                        givens={
                            xj_v: test_set_xj_v
                        },
                        allow_input_downcast=True,
                        name='test_xj_v'
                    )
                    test_xk_v = theano.function(
                        [],
                        outputs=da.output3t_v,
                        givens={
                            xk_v: test_set_xk_v
                        },
                        allow_input_downcast=True,
                        name='test_xk_v'
                    )
                    test_xi_c = theano.function(
                        [],
                        outputs=da.output1_c,
                        givens={
                            xi_c: test_set_xi_c
                        },
                        allow_input_downcast=True,
                        name='test_xi_c'
                    )
                    test_xj_c = theano.function(
                        [],
                        outputs=da.output2t_c,
                        givens={
                            xj_c: test_set_xj_c
                        },
                        allow_input_downcast=True,
                        name='test_xj_c'
                    )
                    test_xk_c = theano.function(
                        [],
                        outputs=da.output3t_c,
                        givens={
                            xk_c: test_set_xk_c
                        },
                        allow_input_downcast=True,
                        name='test_xk_c'
                    )

                    def valid_model(size):
                        vi_v = theano.shared(valid_xi_v()).get_value()
                        vj_v = theano.shared(valid_xj_v()).get_value()
                        vk_v = theano.shared(valid_xk_v()).get_value()
                        vi_c = theano.shared(valid_xi_c()).get_value()
                        vj_c = theano.shared(valid_xj_c()).get_value()
                        vk_c = theano.shared(valid_xk_c()).get_value()
                        vij_v = numpy.dot(vi_v, vj_v)
                        vik_v = numpy.dot(vi_v, vk_v)
                        vij_c = numpy.dot(vi_c, vj_c)
                        vik_c = numpy.dot(vi_c, vk_c)
                        top = []
                        performance = 0.0
                        l = vi_v.tolist()
                        for i in range(size):
                            if(l[i] not in top):
                                top.append(l[i])
                        for i in range(top.__len__()):
                            t = top[i]
                            count1 = 0.0
                            count2 = 0.0
                            for j in range(size):
                                if(l[j] == t):
                                    count1 = count1 + 1
                                    sup = (vij_v[j][j] - vik_v[j][j]) + theta * (vij_c[j][j] - vik_c[j][j])
                                    if (sup > 0):
                                        count2 = count2 + 1
                            performance = performance + float(count2 / count1)

                        return float(performance / top.__len__())

                    def test_model(size):
                        ti_v = theano.shared(test_xi_v()).get_value()
                        tj_v = theano.shared(test_xj_v()).get_value()
                        tk_v = theano.shared(test_xk_v()).get_value()
                        ti_c = theano.shared(test_xi_c()).get_value()
                        tj_c = theano.shared(test_xj_c()).get_value()
                        tk_c = theano.shared(test_xk_c()).get_value()
                        tij_v = numpy.dot(ti_v, tj_v)
                        tik_v = numpy.dot(ti_v, tk_v)
                        tij_c = numpy.dot(ti_c, tj_c)
                        tik_c = numpy.dot(ti_c, tk_c)
                        top = []
                        performance = 0.0
                        l = ti_v.tolist()
                        for i in range(size):
                            if(l[i] not in top):
                                top.append(l[i])
                        for i in range(top.__len__()):
                            t = top[i]
                            count1 = 0.0
                            count2 = 0.0
                            for j in range(size):
                                if(l[j] == t):
                                    count1 = count1 + 1
                                    sup = (tij_v[j][j] - tik_v[j][j]) + theta * (tij_c[j][j] - tik_c[j][j])
                                    if (sup > 0):
                                        count2 = count2 + 1
                            performance = performance + float(count2 / count1)

                        return (float(performance / top.__len__()), tij_v.diagonal(), tij_c.diagonal(), tik_v.diagonal(), tik_c.diagonal())

                    valid_score = []
                    test_score = []
                    train_score = []
                    last_train_cost = []

                    for epoch in range(epoch_time):
                        if epoch>10:
                            learning_rate=0.01

                        cost = 0.0
                        sup_cost = 0.0
                        rec_cost = 0.0
                        mod_cost = 0.0
                        cost_123 = 0.0
                        cost_sqr = 0.0
                        last_cost = 0.0

                        print '%i epoch train start'%(epoch)


                        train_batch_score = []
                        for minibatch_index in range(n_train_batches):
                            minibatch_avg_cost = train_model(minibatch_index)
                            cost = cost + minibatch_avg_cost[0]
                            rec_cost = rec_cost + minibatch_avg_cost[1]
                            sup_cost = sup_cost + minibatch_avg_cost[2]
                            mod_cost = mod_cost + minibatch_avg_cost[3]
                            cost_sqr = cost_sqr + minibatch_avg_cost[4]
                            cost_123 = cost_123 + minibatch_avg_cost[5]
                            d_12v = theano.shared(minibatch_avg_cost[6]).get_value()
                            d_12c = theano.shared(minibatch_avg_cost[7]).get_value()
                            d_13v = theano.shared(minibatch_avg_cost[8]).get_value()
                            d_13c = theano.shared(minibatch_avg_cost[9]).get_value()

                            top = []
                            performance = 0.0
                            l = train_set[0][minibatch_index * batch_size: (minibatch_index + 1) * batch_size].tolist()
                            for i in range(batch_size):
                                if (l[i] not in top):
                                    top.append(l[i])
                            for i in range(top.__len__()):
                                t = top[i]
                                count1 = 0.0
                                count2 = 0.0
                                for j in range(batch_size):
                                    if (l[j] == t):
                                        count1 = count1 + 1
                                        sup = (d_12v[j] - d_13v[j]) + theta * (d_12c[j] - d_13c[j])
                                        if (sup > 0):
                                            count2 = count2 + 1
                                performance = performance + float(count2 / count1)

                            train_batch_score.append(performance / top.__len__())

                        train_score.append(numpy.mean(numpy.asarray(train_batch_score)))
                        print 'train_batch_score',numpy.mean(numpy.asarray(train_batch_score))

                         
                        print('now():' + str(datetime.now()))
                        fi.write('now():' + str(datetime.now()))
                        print 'lamda:%.10f, mu:%.10f, beta:%.10f, theta:%.10f, cost is %.10f, L_rec_cost is %.10f, L_sup_cost is %.10f, L_mod_cost is %.10f, L_sqr is %.10f, L_123 is %.10f ' % (lamda, mu, beta, theta,
                        cost,
                        rec_cost,
                        sup_cost,
                        mod_cost,
                        cost_sqr,
                        cost_123)
                        fi.write(
                        'lamda:%f, mu:%f, %i epochs ended, cost is %f, L_rec_cost is %f, L_sup_cost is %f, L_sqr is %f, L_123 is %f, time %s \n' % (
                        lamda, mu, epoch,
                        cost,
                        rec_cost,
                        sup_cost,
                        cost_sqr,
                        cost_123,
                        str(datetime.now())))

                        fi.flush()
                       



                        print 'train ended'

                        ####################
                        # Validation MODEL #
                        ####################

                        print 'validation start'
                        this_validation_ndcg_score = valid_model(valid_set_size)
                        valid_score.append(this_validation_ndcg_score)
                        print 'validation ended '
                        print 'lamda %.10f, mu %.10f, beta %.10f, theta %.10f, validation score is %.10f' % (iter[0], iter[1], iter[2], iter[3], this_validation_ndcg_score)
                         

                        #########################
                        # TEST MODEL And RECORD #
                        #########################

                        if this_validation_ndcg_score > best_validation_ndcg_score:
                            best_validation_ndcg_score = this_validation_ndcg_score
                            best_valid_iter = iter
                            best_hidden = n_hidden;
                            best_mu=mu;
                            best_lambda=lamda;
                            best_theta=theta;
                            best_beta=beta;

                            ###############
                            # Best Model #
                            #############
                            # numpy.savetxt('w1_v_vt10_100_my.csv', da.W1_v.get_value(), fmt="%.10f")
                            # numpy.savetxt('b1_v_vt10_100_my.csv', da.b1_v.get_value(), fmt="%.10f")
                            # numpy.savetxt('w2_v_vt10_100_my.csv', da.W2_v.get_value(), fmt="%.10f")
                            # numpy.savetxt('b2_v_vt10_100_my.csv', da.b2_v.get_value(), fmt="%.10f")
                            # numpy.savetxt('w1_c_vt10_100_my.csv', da.W1_c.get_value(), fmt="%.10f")
                            # numpy.savetxt('b1_c_vt10_100_my.csv', da.b1_c.get_value(), fmt="%.10f")
                            # numpy.savetxt('w2_c_vt10_100_my.csv', da.W2_c.get_value(), fmt="%.10f")
                            # numpy.savetxt('b2_c_vt10_100_my.csv', da.b2_c.get_value(), fmt="%.10f")
                            # numpy.savetxt('b1_prime_v_vt10_100_my.csv', da.b1_prime_v.get_value(), fmt="%.10f")
                            # numpy.savetxt('b2_prime_v_vt10_100_my.csv', da.b2_prime_v.get_value(), fmt="%.10f")
                            # numpy.savetxt('b1_prime_c_vt10_100_my.csv', da.b1_prime_c.get_value(), fmt="%.10f")
                            # numpy.savetxt('b2_prime_c_vt10_100_my.csv', da.b2_prime_c.get_value(), fmt="%.10f")

                            print 'test start'
                            test_ndcg_score, tij_v, tij_c, tik_v, tik_c = test_model(test_set_size)

                            # numpy.savetxt('./test_mijk/mij_v_' + str(epoch) + '_tv_my.csv', tij_v, fmt="%f")
                            # numpy.savetxt('./test_mijk/mij_c_' + str(epoch) + '_tv_my.csv', tij_c, fmt="%f")
                            # numpy.savetxt('./test_mijk/mik_v_' + str(epoch) + '_tv_my.csv', tik_v, fmt="%f")
                            # numpy.savetxt('./test_mijk/mik_c_' + str(epoch) + '_tv_my.csv', tik_c, fmt="%f")
                            test_score.append(test_ndcg_score)

                            print 'test ended'
                            print 'test_score is %.10f' % (test_ndcg_score)




                        print '%i epoch ended, best_validation_ndcg_score is %f, test_ndcg_score is %f, best valid iteration is lamda %f,mu %f, hidden %f, beta %f, theta %f ' % (
                        epoch, best_validation_ndcg_score,
                        test_ndcg_score, best_valid_iter[0] , best_valid_iter[1] , best_hidden, best_beta, best_theta
                         )
 

    end_time = time.clock()



    print 'running time is %.10f' % (end_time - start_time)
  

if __name__ == '__main__':
    BPR_DAE()
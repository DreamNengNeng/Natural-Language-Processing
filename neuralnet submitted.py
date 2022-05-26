import numpy as np
import argparse
import logging

parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .csv file')
parser.add_argument('validation_input', type=str,
                    help='path to validation input .csv file')
parser.add_argument('train_out', type=str,
                    help='path to store prediction on training data')
parser.add_argument('validation_out', type=str,
                    help='path to store prediction on validation data')
parser.add_argument('metrics_out', type=str,
                    help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')
parser.add_argument('hidden_units', type=int,
                    help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2],
                    help='weight initialization functions, 1: random')
parser.add_argument('learning_rate', type=float,
                    help='learning rate')
parser.add_argument('--debug', type=bool, default=False,
                    help='set to True to show logging')

def args2data(parser):
    """
    Parse argument, create data and label.
    :return:
    X_tr: train data (numpy array)
    y_tr: train label (numpy array)
    X_te: test data (numpy array)
    y_te: test label (numpy array)
    out_tr: predicted output for train data (file)
    out_te: predicted output for test data (file)
    out_metrics: output for train and test error (file)
    n_epochs: number of train epochs
    n_hid: number of hidden units
    init_flag: weight initialize flag -- 1 means random, 2 means zero
    lr: learning rate
    """

    # # Get data from arguments
    out_tr = parser.train_out
    out_te = parser.validation_out
    out_metrics = parser.metrics_out
    n_epochs = parser.num_epoch
    n_hid = parser.hidden_units
    init_flag = parser.init_flag
    lr = parser.learning_rate

    X_tr = np.loadtxt(parser.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr[:, 0] = 1.0 #add bias terms

    X_te = np.loadtxt(parser.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te[:, 0]= 1.0 #add bias terms

    return (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,
            n_epochs, n_hid, init_flag, lr)

def shuffle(X, y, epoch):
    """
    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]

def random_init(shape):
    """
    Randomly initialize a numpy array of the specified shape
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    # DO NOT CHANGE THIS
    np.random.seed(np.prod(shape))   #np.prod() is numpy mathematical library method that returns the product of the array of elements over a given axis.
    weights =  np.random.uniform(-0.1, 0.1, shape)

    return weights 

def zero_init(shape):
    """
    Initialize a numpy array of the specified shape with zero
    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    weights = np.zeros(shape, dtype = int)
    return weights


class NN(object):
    def __init__(self, lr, n_epoch, weight_init_fn, input_size, hidden_size, output_size):
        """
        Initialization
        :param lr: learning rate
        :param n_epoch: number of training epochs
        :param weight_init_fn: weight initialization function
        :param input_size: number of units in the input layer
        :param hidden_size: number of units in the hidden layer
        :param output_size: number of units in the output layer
        """

        self.lr = lr
        self.n_epoch = n_epoch
        self.weight_init_fn = weight_init_fn
        self.n_input = input_size
        self.n_hidden = hidden_size
        self.n_output = output_size

        # initialize weights and biases for the models
        # HINT: pay attention to bias here
        self.w1 = weight_init_fn([hidden_size, input_size])
        self.w2 = weight_init_fn([output_size, hidden_size+1])

        # initialize parameters for adagrad
        self.epsilon = np.array([1e-5])
        self.grad_sum_w1 = 0
        self.grad_sum_w2 = 0
        

        
        input_size = X_tr.shape[1]  #get the input feature, shape[1] gives us # of columns
        
        # initialize weights and biases for the models
        # HINT: pay attention to bias here
        # the dim below is determined based on the input size, hidden size, and output size
        
        shape1 = (hidden_size, input_size)
        shape2 = (output_size, hidden_size + 1)
        # initialize w1(alpha), w2(beta)
        if init_flag == 1:
            w1 = random_init(shape1)
            w1[:,0] = 0
            w2 = random_init(shape2)
            w2[:,0] = 0
        else:
            w1 = zero_init(shape1)
            w2 = zero_init(shape2)


        
      
def print_weights(nn):
    """
    An example of how to use logging to print out debugging infos.
    Note that we use the debug logging level -- if we use a higher logging
    level, we will log things with the default logging configuration,
    causing potential slowdowns.

    Note that we log NumPy matrices on separate lines -- if we do not do this,
    the arrays will be turned into strings even when our logging is set to
    ignore debug, causing potential massive slowdowns.
    :param nn: your model
    :return:
    """
    logging.debug(f"shape of w1: {nn.w1.shape}")
    logging.debug(nn.w1)
    logging.debug(f"shape of w2: {nn.w2.shape}")
    logging.debug(nn.w2)

def forward(X, nn):
    """
    Neural network forward computation.
    Follow the pseudocode!
    :param X: input data
    :param nn: neural network class
    :return: output probability
    """
    nn.a = np.dot(nn.w1, X)
    nn.z = 1/(1+np.exp(-nn.a))
    # add a bia for z
    nn.z[:,0] = 1.0
    b = np.dot(nn.w2, nn.z)
    #y_hat = softmax(b)
    y_hat = np.exp(b) / (np.sum(np.exp(b)))
    return y_hat

def backward(X, y, y_hat, nn):
    """
    Neural network backward computation.
    Follow the pseudocode!
    :param X: input data
    :param y: label
    :param y_hat: prediction
    :param nn: neural network class
    :return:
    d_w1: gradients for w1
    d_w2: gradients for w2
    """
   # Place intermediate quantities x,z,z,b,y_hat in scope
    g_b = y_hat - y
    g_beta, g_z = np.dot(g_b, nn.z.T), np.dot(nn.w2.T, g_b)
    temp = np.multiply(nn.z, 1-nn.z)
    g_a = np.multiply(temp, g_z)[1:,:]
    g_alpha= np.dot(g_a, X.T)
    return (g_alpha, g_beta)
 

def SGD(X_tr, y_tr, nn):
    """
    Train the network using SGD for some epochs.
    :param X_tr: train data
    :param y_tr: train label
    :param nn: neural network class
    """
    y_hot = np.eye(10)[y_tr]
    J_train_result, J_val_result = [], []
    
    for k in range(nn.n_epoch):
        for i in range(len(X_tr)):
            x_i = X_tr[i].reshape(-1,1)
            y_i = y_hot[i].reshape(-1,1)
            
            y_hat_i = forward(x_i, nn)
            (g_alpha, g_beta) = backward(x_i,y_i, y_hat_i, nn)
            

            # adagrad update g_alpha and g_beta
            nn.grad_sum_w1 = nn.grad_sum_w1 + np.multiply(g_alpha,g_alpha)
            nn.grad_sum_w2 = nn.grad_sum_w2 + np.multiply(g_beta,g_beta)
            nn.w1 = nn.w1 - nn.lr/np.sqrt(nn.grad_sum_w1+ nn.epsilon)*g_alpha
            nn.w2 = nn.w2 - nn.lr/np.sqrt(nn.grad_sum_w2 + nn.epsilon)*g_beta
            
        #calculate the cross entropy on train and validation data set
        y_pre_train = predict(X_tr,y_tr, nn)
        y_pre_val = predict(nn.X_te, nn.y_te, nn)
        J_train = -np.sum(np.multiply(y_tr, np.log(y_pre_train)))/len(y_tr)
        J_val = -np.sum(np.multiply(nn.y_te, np.log(y_pre_val)))/len(nn.y_te)
        J_train_result.append(J_train)
        J_val_result.append(J_val)
        
    return J_train_result, J_val_result

def predict(X, y, nn):
    """
    Compute the label and error rate.
    :param X: input data
    :param y: label
    :param nn: neural network class
    :return:
    labels: predicted labels
    error_rate: prediction error rate
    """
    train_y = nn.SGD(X)
    error_rate = np.sum(train_y != y)/len(y)
    return train_y, error_rate



if __name__ == "__main__":

    args = parser.parse_args()
    if args.debug:  
        logging.basicConfig(format='[%(asctime)s] {%(pathname)s:%(funcName)s:%(lineno)04d} %(levelname)s - %(message)s',
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)
    logging.debug('*** Debugging Mode ***')
    # Note: You can access arguments like learning rate with args.learning_rate

    # initialize training / test data and labels
    out_tr = args.train_out
    out_te = args.validation_out
    out_metrics = args.metrics_out
    n_epoch = args.num_epoch
    weight_init_fn = args.weight_init_fn
    init_flag = args.init_flag
    lr = args.learning_rate
    input_size = args.n_input
    hidden_size = args.n_hidden
    output_size = args.n_output


    X_tr = np.loadtxt(args.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr[:, 0] = 1.0 #add bias terms

    X_te = np.loadtxt(args.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te[:, 0]= 1.0 #add bias terms

    # Build model
    my_nn = NN(lr, n_epoch, weight_init_fn, input_size, hidden_size, output_size)

    # train model
    J_train_result, J_val_result = SGD(X_tr,y_tr,my_nn)
    # test model and get predicted labels and errors
    labels, errors = predict(X_tr, y_tr, my_nn)
    # write predicted label and error into file

    result = []
    result.append( "error(train): " + str(predict(X_tr, y_tr, my_nn)) )
    result.append( "error(validaton): " + str(predict(X_te, y_te, my_nn)) )

    np.savetxt(args.train_out, J_train_result, fmt='%s', delimiter ='\t')
    np.savetxt(args.validation_out, J_val_result, fmt='%s', delimiter ='\t')
    np.savetxt(args.metrics_out, result, fmt='%s', delimiter ='\t')





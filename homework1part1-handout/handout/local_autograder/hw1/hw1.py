"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.
Notes:
The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.
# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import numpy as np
import os
import math
class Activation(object):

    """
    Interface for activation functions (non-linearities).
    In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an abstract base class for the others

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):
    """
    Sigmoid non-linearity
    """
    # Remember do not change the function signatures as those are needed to stay the same for AL

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        self.state=np.where(x >= 0, 
                         1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))
        return  np.where(x >= 0, 
                         1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))

    def derivative(self):
        return self.state * (1 - self.state)


class Tanh(Activation):

    """
    Tanh non-linearity
    """
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state=np.tanh(x)
        return np.tanh(x)

    def derivative(self):
        return 1.0 -(self.state)**2


class ReLU(Activation):

    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
      self.state=np.maximum(0,x)
      return np.maximum(0,x)

    def derivative(self):
      self.state[self.state>0]=1
      return self.state

# Ok now things get decidedly more interesting. The following Criterion class
# will be used again as the basis for a number of loss functions (which are in the
# form of classes so that they can be exchanged easily (it's how PyTorch and other
# ML libraries do it))


class Criterion(object):

    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


class SoftmaxCrossEntropy(Criterion):

    """
    Softmax loss
    """

    def __init__(self):

        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None
        self.eps=1e-8
        
    def forward(self, x, y):
        self.logits = x
        self.labels = y.astype(int)
        N=self.logits.shape[0]
        exps = np.exp(self.logits-self.eps)
        self.sm=exps / np.exp(np.log(np.sum(exps,axis=1).reshape(-1,1))+self.eps)
        log_probs=np.log(self.sm)
        ce = -log_probs[np.arange(N), np.argmax(self.labels,axis=1)]
        return ce

    def derivative(self):
        N=self.logits.shape[0]
        dx = self.sm
        dx=dx-self.labels
        #dx[np.arange(N), np.argmax(self.labels,axis=1)]-=1
        return  dx


class BatchNorm(object):

    def __init__(self, fan_in, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        if eval:
            x_norm=(x-self.running_mean)/np.sqrt(self.running_var+self.eps)
            self.out=self.gamma*x_norm+self.beta
            print(self.out)
        else:
            self.x = x
            self.mean=np.mean(x,axis=0)
            self.var = np.var(x,axis=0)
            self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * self.mean
            self.running_var = self.alpha * self.running_var + (1 - self.alpha) * self.var
            self.norm=(self.x-self.mean)/np.sqrt(self.var+self.eps)
            self.out = self.norm*self.gamma+self.beta
        return self.out

    def backward(self, delta):
        X_mu=self.x-self.mean
        N=X_mu.shape[0]
        std_inv = 1. / np.sqrt(self.var + self.eps)
        dnorm=delta*self.gamma
        dvar = np.sum(dnorm * X_mu, axis=0) * -.5 * std_inv**3
        dmu = np.sum(dnorm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)
        dx = (dnorm * std_inv) + (dvar * 2 * X_mu /N) + (dmu / N)
        self.dgamma = np.sum(delta * self.norm, axis=0)
        self.dbeta = np.sum(delta, axis=0)
        return dx


# These are both easy one-liners, don't over-think them
def random_normal_weight_init(d0, d1):
    return np.random.normal(size=(d0,d1))


def zeros_bias_init(d):
    return np.zeros((1,d))


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.hiddens=hiddens
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------

        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        self.list_layers=[self.input_size]+self.hiddens+[self.output_size]
        self.items=[]
        self.W = [weight_init_fn(self.list_layers[i],self.list_layers[i+1]) for i in range(len(self.list_layers)-1)]
        self.dW = [np.zeros((self.list_layers[i],self.list_layers[i+1])) for i in range(len(self.list_layers)-1)]
        self.b =  [bias_init_fn((self.list_layers[i+1])) for  i in range(len(self.list_layers)-1)]
        self.db = [bias_init_fn((self.list_layers[i+1])) for  i in range(len(self.list_layers)-1)]
        
        
        self.d_vW = [np.zeros((self.list_layers[i],self.list_layers[i+1])) for i in range(len(self.list_layers)-1)]
        self.d_vb =  [bias_init_fn((self.list_layers[i+1])) for  i in range(len(self.list_layers)-1)]
        
        # HINT: self.foo = [ bar(???) for ?? in ? ]

        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = [BatchNorm(1) for i in range(self.num_bn_layers)]


        # Feel free to add any other attributes useful to your implementation (input, output, ...)

    def forward(self, x):
        a=x    
        for i in range(self.nlayers):
            self.items.append(a)
            item=np.dot(a,self.W[i])+self.b[i]
            if self.bn and i<self.num_bn_layers: 
                item=self.bn_layers[i].forward(item, 
                                   not(self.train_mode)
                                   )
            a=self.activations[i](item)
        self.output=a
        return self.output

    def zero_grads(self):
        self.dW = [np.zeros((self.list_layers[i],self.list_layers[i+1])) for i in range(len(self.list_layers)-1)]
        self.db = [zeros_bias_init((self.list_layers[i+1])) for  i in range(len(self.list_layers)-1)]
        return

    def step(self):
        if self.momentum==0:
            self.W=[self.W[i]-self.lr*self.dW[i] for i in range(len(self.W))]
            self.b=[self.b[i]-self.lr*self.db[i] for i in range(len(self.b))]
        else:
            self.d_vW=[self.momentum*self.d_vW[i] - self.lr*self.dW[i] for i in range(len(self.W))]
            self.W=[self.W[i]+self.d_vW[i] for i in range(len(self.W))]
            
            self.d_vb=[self.momentum*self.d_vb[i] - self.lr*self.db[i] for i in range(len(self.b))]
            self.b=[self.b[i]+self.d_vb[i] for i in range(len(self.b))]
        if self.bn:
            for i in range(len(self.bn_layers)):
                self.bn_layers[i].gamma= self.bn_layers[i].gamma-self.bn_layers[i].dgamma
                self.bn_layers[i].beta= self.bn_layers[i].beta-self.bn_layers[i].dbeta
                
        return

    def backward(self, labels):
        self.criterion(self.output,labels)
        dout=self.criterion.derivative()
        N=dout.shape[0]
        for i in reversed(range(self.nlayers)):
            dout=np.multiply(dout, self.activations[i].derivative())   
            if self.bn and i<self.num_bn_layers: 
                dout=self.bn_layers[i].backward(dout)
            self.dW[i]=np.dot(self.items.pop().T,dout)/N
            self.db[i]=np.sum(dout,axis=0)/N
            dout=np.dot(dout,self.W[i].T)
        return

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, test = dset
    trainx, trainy = train
    valx, valy = val
    testx, testy = test

    idxs = np.arange(len(trainx))

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []

    # Setup ...

    for e in range(nepochs):

        # Per epoch setup ...

        for b in range(0, len(trainx), batch_size):

            pass  # Remove this line when you start implementing this
            # Train ...

        for b in range(0, len(valx), batch_size):

            pass  # Remove this line when you start implementing this
            # Val ...

        # Accumulate data...

    # Cleanup ...

    for b in range(0, len(testx), batch_size):

        pass  # Remove this line when you start implementing this
        # Test ...

    # Return results ...

    # return (training_losses, training_errors, validation_losses, validation_errors)

    raise NotImplemented
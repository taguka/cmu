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
            x_norm=(self.x-self.running_mean)/np.sqrt(self.running_var+self.eps)
            self.out=self.gamma*x_norm+self.beta
        else:
            self.x = x
            self.mean = np.mean(self.x)
            self.var = np.mean((self.x-np.mean(self.x))**2)
            self.norm = (self.x-self.mean)/np.sqrt(self.var+self.eps)
            self.out = self.norm*self.alpha+self.gamma
            step_1=np.mean(self.x)
            print('step_1',step_1.shape)
            step_2=self.x-step_1
            print('step_2',step_2.shape)
            step_3=np.square(step_2)
            print('step_3',step_3.shape)
            step_4=np.mean(step_3)
            print('step_4',step_4.shape)
            step_5=np.sqrt(step_4+self.eps)
            print('step_5',step_5.shape)
            step_6=1/step_5
            print('step_6',step_6.shape)
            step_7=step_2*step_6
            print('step_7',step_7.shape)
            step_8=step_7*self.gamma
            print('step_8',step_8.shape)
            step_9=step_8+self.beta
            print('step_9',step_9.shape)
            self.out=step_9
            self.cache=(step_8, step_7, step_6, step_5, step_4, step_3, step_2, step_1)
            self.running_mean = 0.9 * self.running_mean + (1 - 0.9) * self.mean
            self.running_var = 0.9 * self.running_var + (1 - 0.9) * self.var
        return self.out

    def backward(self, delta):
        step_8, step_7, step_6, step_5, step_4, step_3, step_2, step_1 = self.cache
        self.dbeta=np.sum(delta,axis=0)
        dstep_8=delta
        print('dstep_8',dstep_8.shape)
        self.dgamma=np.sum(dstep_8*step_7)
        dstep_7=dstep_8*self.gamma
        dstep_2_1=dstep_7*step_6
        dstep_6=np.sum(dstep_7*step_2)
        dstep_5=dstep_6*(-1/np.square(step_5))
        dstep_4=dstep_5*(1/(2*np.sqrt(step_4+self.eps)))
        dstep_3=(dstep_4)
        dstep_2_2=dstep_3*2*step_2
        dstep_1=np.sum((dstep_2_1+dstep_2_2), axis=0)*(-1)
        dx_1=(dstep_2_1+dstep_2_2)*1
        dx_2=dstep_1
        dx=dx_1+dx_2
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
                item=self.bn_layers[i](item)
            a=self.activations[i](item)
        self.output=a
        return self.output

    def zero_grads(self):
        self.dW = [np.zeros((self.list_layers[i],self.list_layers[i+1])) for i in range(len(self.list_layers)-1)]
        self.db = [zeros_bias_init((self.list_layers[i+1])) for  i in range(len(self.list_layers)-1)]
        return

    def step(self):
        self.W=[self.W[i]-self.lr*self.dW[i] for i in range(len(self.W))]
        self.b=[self.b[i]-self.lr*self.db[i] for i in range(len(self.b))]
        return

    def backward(self, labels):
        self.criterion(self.output,labels)
        dout=self.criterion.derivative()
        N=dout.shape[0]
        for i in reversed(range(self.nlayers)):
            if self.bn and i<self.num_bn_layers: 
                dout=self.bn_layers[i].backward(dout)
            dout=np.multiply(dout, self.activations[i].derivative())             
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
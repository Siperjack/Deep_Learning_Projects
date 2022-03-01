import numpy as np
import random as ran

###This should set the inputlayer of the network as the minibatch C, and then propegate by F(W@C + bias) one layer at the time
### NB: Only X is changed ever in this function. The X of the last layer shoult be the output of the network

def ReLU(x):
    return x * (x > 0)

def DReLU(x):
    return 1 * (x >= 0)

def tanh(x):
    return np.tanh(x)

def Dtanh(x):
    return 1 - np.tanh(x)**2


def soft_max(x): # matrix
    return np.exp(x) / (np.exp(x).sum(axis = 1, keepdims = True)) # shape is Cxn
    
def stable_soft_max(x): # matrix
    # print("output of softmax: ", np.exp(x) / np.exp(x).sum(axis = 1, keepdims = True))
    z = np.copy(x)
    for i, xi in enumerate(x):
        z[i] = np.exp(xi - np.max(xi)) / (np.exp(xi - np.max(xi)).sum()) # shape is Cxn
    return z

def Dsoft_max(x):#input should be the softmaxed x
    z = x
    # print("Dsoftmaxshape is: ", np.shape(z * (1 - z)))
    C = len(x)
    n = len(x[0,:])
    Jac_softmax = np.zeros((C,n,n))
    for i in range(C):
        for j in range(n):
            for k in range(n):
                if j == k:
                    Jac_softmax[i,j,k] = z[i,j] * (1 - z[i,k])
                else:
                    Jac_softmax[i,j,k] = - z[i,j] * z[i,k]
    return Jac_softmax #shape is Cxnxn

def sigmoid(x):
    return 1/(1 + np.exp(x))

def Dsigmoid(x):
    return x * (1 - x)

def CrossEntropy(x,y):
    return (-y*np.log(x)).sum(axis = 1)

def DCrossEntropy(x,y):
    C = len(x)
    n = len(x[0,:])
    grad = np.zeros((C,n))
    for i in range(C):
        grad[i] = -y[i]/(x[i])
    return grad #shape is Cxn

def DCrossEntropySoftMax(z,y):
    return z - y
    
    
def DCrossEntropySoftMax1(z,x,y):
    C = len(z)
    n = len(z[0,:])
    grad = np.zeros((C,n))
    for i in range(C):
        k = np.argmax(y[i])
        for j in range(n):
            if k == j:
                grad[i,j] = -y[i,k]/z[i,k] * x[i,k] * (1 - x[i,k])
            else:
                grad[i,j] = y[i,k]/z[i,k] * x[i,k] * x[i,j]
    return grad #shape is Cxn
        

def L2reg(network):
    zum = 0
    for l in network.layers:
        zum += np.einsum("ij -> ", l.W)**2/2
    return zum

def DL2reg(W):
    return W

def L1reg(network):
    zum = 0
    for l in network.layers:
        zum += np.abs(np.einsum("ij -> ", l.W))
    return zum

def DL1reg(W):
    return 1 * (W > 0)

def SSE(x, y):
    a = np.zeros(np.shape(x[:,0]))
    for i in range(len(x)):
        a[i] = np.linalg.norm(x[i] - y[i])**2/2
    return a #shape is C
def DSSE(x, y):
    a = np.zeros(np.shape(x[:,:]))
    for i in range(len(x)):
        a[i] = x[i] - y[i]
    return a


func_dic = {
    "sse": SSE,
    "crossentropy": CrossEntropy,
    "relu": ReLU,
    "softmax": soft_max,
    "stablesoftmax": stable_soft_max,
    "l1": L1reg,
    "l2": L2reg,
    "tanh": tanh,
    "sigmoid": sigmoid

}
Dfunc_dic = {
    "sse": DSSE,
    "crossentropy": DCrossEntropy,
    "relu": DReLU,
    "softmax": Dsoft_max, 
    "stablesoftmax": Dsoft_max,
    "l1": DL1reg,
    "l2": DL2reg,
    "tanh": Dtanh,
    "sigmoid": Dsigmoid
}


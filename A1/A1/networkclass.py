import numpy as np
import functions
from layerclass import *

###Probably want to initialise the network once. This is supposed to be done with a textfile. 

class network():
    def __init__(self, input_size, file):
        self.file = file
        self.width = [input_size]
        self.layers = []
        self.training_performance = []
        self.training_hitrate = []
        self.validation_performance = []
        self.validation_hitrate = []
        with open(file, "r") as f:
            content = []
            for line in f:
                if line.startswith("#"):
                    continue
                else: 
                    line = line.strip("\n")
                    line = line.split(",") 
                    content.append(line)
        self.depth = len(content) - 2 #amount of hidden layers
        self.lf = functions.func_dic[content[0][0].lower()]
        self.Dlf = functions.Dfunc_dic[content[0][0].lower()]
        self.af = functions.func_dic[content[0][1].lower()]
        self.Daf = functions.Dfunc_dic[content[0][1].lower()]
        self.lr = float(content[0][2])
        self.rval = float(content[0][3])
        self.rf = functions.func_dic[content[0][4].lower()]
        self.Drf = functions.Dfunc_dic[content[0][4].lower()]
        self.output_afunc = functions.func_dic[content[-1][0].lower()]
        self.output_Dafunc = functions.Dfunc_dic[content[-1][0].lower()]
        self.output = None
                ##Defaults
        self.layer_brs = ["(0:1)"] * (self.depth)
        self.layer_wrs = ["gr"] * (self.depth)
        self.layer_lrs = [self.lr] * (self.depth)
        self.layer_afs = [self.af] * (self.depth)
        self.layer_Dafs = [self.Daf] * (self.depth)
                ##Custom layer
        for i, layer_cont in enumerate(content):
            if i == 0 or i == self.depth + 2:
                continue
            for j, string in enumerate(layer_cont):
                if string == "af":
                    self.layer_afs[i-1] = functions.func_dic[layer_cont[j + 1]]
                    self.layer_Dafs[i-1] = functions.Dfunc_dic[layer_cont[j + 1]]
                if string == "wr":
                    self.layer_wrs[i-1] = layer_cont[j + 1]
                if string == "br":
                    self.layer_brs[i-1] = layer_cont[j + 1]
                if string == "lr":
                    self.layer_lrs[i-1] = float(layer_cont[j + 1])
                
                
                ##None-defaults
        for i in range(self.depth):
            self.width.append(int(content[i+1][0]))

        for i in range(self.depth):
            self.layers.append(layer(self.width[i+1], 
            self.width[i], [self.layer_afs[i], self.layer_Dafs[i]], self.layer_lrs[i], 
            self.layer_brs[i], self.layer_wrs[i], self.rval, [self.rf, self.Drf]))
        #Initialise child class variable on all layers
        for l in range(self.depth - 1):
            self.layers[l].child = self.layers[l + 1]
            self.layers[l + 1].parent = self.layers[l]
        #print("number of layers are: ", len(self.layers))

    def uppdateParameter(self):
                for l in self.layers:
                    if len(l.dW[0]) == len(l.W[0]):
                        deltaW = l.dW
                        deltab = l.db
                    else:
                        deltaW = l.dW.sum(axis = 0, keepdims = False)
                        deltab = l.db.sum(axis = 0, keepdims = True).transpose()
                    l.W = l.W - self.lr*(deltaW + self.rval*self.Drf(l.W))
                    l.b = l.b - self.lr*deltab
    def propagate_forward(self, x, y, hitrate = False, performance = False, saveVals = True):
        
        for l in self.layers:
            x = l.Forward(x)
        x = self.layers[-1].X
        self.output = self.output_afunc(self.layers[-1].X)
        if saveVals:
            z = self.output
            x = self.layers[-1].X
            loss_sum = self.lf(z, y) + self.rval*self.rf(self)
            predict = self.isRight(z, y)/len(z)
            hitrate.append(predict)
            performance.append(loss_sum.sum(axis = 0))

    def propagate_backward(self, y):
        z = self.output
        x = self.layers[-1].X
        if self.lf == functions.CrossEntropy:
            Jac_LX = functions.DCrossEntropySoftMax(z, y)
        else:
            Jac_LS = self.Dlf(z, y) # C x n
            Jac_SX = self.output_Dafunc(x) # C x n x n
            Jac_LX = np.einsum("Ci,Cji->Cj", Jac_LS, Jac_SX)
        for l in reversed(self.layers):
            Jac_LX = l.Backward(Jac_LX) #should matrix multiply each of the C number of jacobians and produce the C results in a matrix
    
        


    ### Picks out random minibatches and trains(forward + backward prop) for the given amount of epochs, with uppdating of weighthappening
    ### only after each epoch.
    def train_network_with_SGD(self, S, C, V, epochs = 10):#S = training set, C = size of minibatch
        epoch = len(S)//C
        for i in range(epochs):
            for j in range(epoch):
                np.random.shuffle(S)
                np.random.shuffle(V)
                minibatch = S[0:C]
                minibatch_input = np.zeros((C, len(S[0].flattened_image)))
                minibatch_sol = np.zeros((C, len(S[0].sol)))
                minibatch_v = V[0:C]
                minibatch_input_v = np.zeros((C, len(V[0].flattened_image)))
                minibatch_sol_v = np.zeros((C, len(V[0].sol)))
                for k in range(C):
                    minibatch_input[k] = minibatch[k].flattened_image
                    minibatch_sol[k] = minibatch[k].sol
                    minibatch_input_v[k] = minibatch_v[k].flattened_image
                    minibatch_sol_v[k] = minibatch_v[k].sol
                self.propagate_forward(minibatch_input_v, minibatch_sol_v, self.validation_hitrate,self.validation_performance)
                self.propagate_forward(minibatch_input, minibatch_sol, self.training_hitrate,self.training_performance) #fills the network with values for X
                self.propagate_backward(minibatch_sol) #returns delta W's for minibatch C
                self.uppdateParameter() #uppdate after every batch
    
    def isRight(self,x,y):
        hitrate = 0
        for i, xi in enumerate(x):
            if y[i,np.argmax(xi)] == 1:
                hitrate +=1
        return hitrate/len(x)
    
    def test(self, T):
        C = len(T)
        testset_input = np.zeros((C, len(T[0].flattened_image)))
        testset_sol = np.zeros((C, len(T[0].sol)))
        for k in range(C):
                testset_input[k] = T[k].flattened_image
                testset_sol[k] = T[k].sol
        x = testset_input
        y = testset_sol
        self.propagate_forward(x, y, saveVals = False)
        z = self.output
        loss_sum = self.lf(z, y) + self.rval*self.rf(self)
        predict = self.isRight(z,y)
        rightwrong_list = self.sortRight(z, y, T)
        return loss_sum, predict, rightwrong_list

    def sortRight(self, x, y, T):
        rightlist = []
        wronglist = []
        for i, xi in enumerate(x):
            if y[i,np.argmax(xi)] == 1:
                rightlist.append(T[i])
            else:
                wronglist.append([T[i], np.argmax(x[i])])
        return[rightlist, wronglist]
                
                



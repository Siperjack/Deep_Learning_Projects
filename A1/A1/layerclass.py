import numpy as np
import functions



class layer():
    def __init__(self, width, prevwidth, activ_func, personal_learningrate, 
    br, wr, regval, regfunc):
        self.parent = None
        self.child = None
        self.prevwidth = prevwidth
        self.prevX = None
        self.lr = personal_learningrate
        self.width = width
        self.afunc = activ_func[0]
        self.Dafunc = activ_func[1]
        self.b = self.init_b(br,(width, 1))# n x 1
        self.W = self.init_W(wr,(width, prevwidth)) # n x m
        self.X = None # n x m @ m x C
        self.dW = None
        self.db = None
        self.first_iter = True
        self.rval = regval
        self.rf = regfunc[0]
        self.Drf = regfunc[1]
        
    def init_b(self, instruc_string, shape):
        if instruc_string[0] == "(":
            collonIndex = instruc_string.find(":")
            a = float(instruc_string[1:collonIndex])
            b = float(instruc_string[collonIndex + 1:len(instruc_string) - 1])
        
        return np.random.uniform(a, b, shape)
    
    def init_W(self, instruc_string, shape):
        if instruc_string[0] == "(":
            collonIndex = instruc_string.find(":")
            a = float(instruc_string[1:collonIndex])
            b = float(instruc_string[collonIndex + 1:len(instruc_string) - 1])
        if instruc_string == "gr":
            b = 1/np.sqrt(shape[1])
            a = -b
        return np.random.uniform(a, b, shape)
            
    def Forward(self, x):
        self.prevX = x
        self.U = np.einsum("Ci, ji -> Cj", x, self.W)
        B =  np.tile(self.b, len(self.U)).transpose()
        self.X = self.afunc(self.U + B)
        return self.X
    
    def Backward(self, upstream_Jac): #We assume here all hidden layer activation functions jacobians does not have non-zero cross terms
        C = len(self.X[:])
        n = self.width
        m = self.prevwidth
        JacW = np.einsum("Ci,Cj->Cij", self.Dafunc(self.X), self.prevX)
        self.dW = np.einsum("Ci, Cij->ij", upstream_Jac, JacW)
        Jacb = np.einsum("Ci->i", upstream_Jac)
        self.db = np.array([Jacb]).transpose()
        JacX = np.einsum("Ci,ij->Cij", self.Dafunc(self.X), self.W)
        downstream_Jac = np.einsum("Ci,Cij->Cj", upstream_Jac, JacX)
        return downstream_Jac

    def Backward1(self, upstream_Jac): #We assume here all hidden layer activation functions jacobians does not have non-zero cross terms
        C = len(self.X[:])
        n = self.width
        m = self.prevwidth
        JacW = np.zeros((C,n,m))
        JacW = np.einsum("Ci,Cj->Cij", (upstream_Jac * self.Dafunc(self.X)), self.prevX)
        self.dW = JacW
        Jacb = np.zeros((C,n))
        Jacb = np.einsum("Ci,C->Ci", (upstream_Jac * self.Dafunc(self.X)), np.ones(C))
        self.db = Jacb
        downstream_Jac = np.zeros((C,m))
        downstream_Jac = np.einsum("ij,Ci->Cj", self.W, (upstream_Jac * self.Dafunc(self.U)))
        return downstream_Jac
import numpy as np
np.random.seed(42)


class perceptron():
    def __init__(self,dim,outputdim) :
        self.dim = dim
        self.outputdim = outputdim
        w1 = np.random.randn(dim, 100)  # 100是隐藏节点的个数，可根据需要调整
        b1 = np.random.randn(100)
        w2 = np.random.randn(100, outputdim)
        b2 = np.random.randn(outputdim)
        self.parameter = [w1,b1,w2,b2]
    def logistic(self,x):
        epsilon = 0.00001
        sum = np.exp(x).sum(axis=1).reshape(-1,1)
        return np.exp(x)/sum 
    def relu(self,x):
        return x.clip(0)
    def forward(self,x):
        x = x@self.parameter[0] + self.parameter[1]
        x = self.relu(x)
        x = x@self.parameter[2] + self.parameter[3]
        x = self.relu(x)
        x = self.logistic(x)
        return x

class loss():
    def __init__(self,model) :
        self.model=model
        self.grad = None 
        self.input = None
        self.label = None
    def __call__(self, label, x) :
        batch = label.shape
        out=self.model.forward(x)
        out = -np.log(out)/batch
        self.input = x
        self.label = label
        self.output = np.array([out[i][label[i]] for i in range(label.size)]).sum()
        return self.output
    def backward(self):
        dx = 0.00001
        t = []
        for i in self.model.parameter:
            grad = np.zeros_like(i)
            for j in range(i.size):
                i.reshape(-1)[j]+=dx
                batch = self.label.size
                out=self.model.forward(self.input)
                i.reshape(-1)[j]-=dx
                out = -np.log(out)/batch
                dy=np.array([out[i][self.label[i]] for i in range(batch)]).sum()-self.output
                dy_dx = dy/dx
                grad.reshape(-1)[j]=dy_dx
            t.append(grad)
        return t
if __name__ == "__main__":
    model = perceptron(28*28,10)
    Loss = loss(model)
    x = np.random.randn(10,28*28)
    label = np.ones(3,dtype= int)
    lr = 0.01


    for i in range(100):
        print( Loss(label,x))
        grad=Loss.backward()
        # print(grad)
        for j in range(4):
            model.parameter[j] -= lr*grad[j]


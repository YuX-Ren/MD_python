import numpy as np

class Perceptron():
    def __init__(self, dim, outputdim):
        self.dim = dim
        self.outputdim = outputdim
        w1 = np.random.randn(dim, 100)
        b1 = np.random.randn(100)
        w2 = np.random.randn(100, outputdim)
        b2 = np.random.randn(outputdim)
        self.parameter = [w1, b1, w2, b2]

    def softmax(self, x):
        x_exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return x_exp / np.sum(x_exp, axis=1, keepdims=True)

    def relu(self, x):
        return np.maximum(x, 0)

    def forward(self, x):
        x = x @ self.parameter[0] + self.parameter[1]
        x = self.relu(x)
        x = x @ self.parameter[2] + self.parameter[3]
        x = self.softmax(x)
        return x

class Loss():
    def __init__(self, model):
        self.model = model
        self.grad = None
        self.input = None
        self.label = None

    def __call__(self, label, x):
        batch = label.shape[0]
        out = self.model.forward(x)
        self.input = x
        self.label = label
        epslion = 0.000001
        log_likelihood = -np.log(out[range(batch), label]+epslion)
        return np.mean(log_likelihood)

    def backward(self):
        batch = self.label.shape[0]
        out = self.model.forward(self.input)
        out[range(batch), self.label] -= 1
        out /= batch

        d_relu = (self.input @ self.model.parameter[0] + self.model.parameter[1] > 0).astype(float)

        d_w2 = np.dot(self.model.relu(self.input @ self.model.parameter[0] + self.model.parameter[1]).T, out)
        d_b2 = np.sum(out, axis=0)

        d_w1 = np.dot(self.input.T, np.dot(out, self.model.parameter[2].T) * d_relu)
        d_b1 = np.sum(np.dot(out, self.model.parameter[2].T) * d_relu, axis=0)

        return [d_w1, d_b1, d_w2, d_b2]
if __name__ == "__main__":
    model = Perceptron(28*28,10)
    loss = Loss(model)
    x = np.random.randn(10,28*28)
    label = np.ones(3,dtype= int)
    lr = 0.01


    for i in range(100):
        print( loss(label,x))
        grad=loss.backward()
        # print(grad)
        for j in range(4):
            model.parameter[j] -= lr*grad[j]


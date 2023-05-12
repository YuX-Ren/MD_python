import numpy as np
from IMDb_dataset import get_data
class Transformer():
    def __init__(self, dim, outputdim,seq_len,vocabulary_size):
        self.dim = dim
        self.outputdim = outputdim
        self.seq_len=seq_len
        self.heads = 1
        attention_q = np.random.randn(dim, dim)
        attention_k = np.random.randn(dim, dim)
        attention_v = np.random.randn(dim, dim)
        oproj = np.random.randn(dim, dim)
        embedding_layer = np.random.randn(vocabulary_size, dim)
        w1 = np.random.randn(dim, dim*2)
        w2 = np.random.randn(dim*2, 2)
        self.parameter = [w1,  w2, attention_q,attention_k,attention_v,oproj,embedding_layer]
    def softmax(self, x,axis):
        x_exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return x_exp / np.sum(x_exp, axis=axis, keepdims=True)

    def multihead_attention(self,x):
        batch_size,seq_length,_ = x.shape
        d_k = int(self.dim/self.heads)
        self.q = (x@self.parameter[2]).reshape(batch_size, seq_length, self.heads, d_k).transpose(0, 2, 1, 3)
        self.k = (x@self.parameter[3]).reshape(batch_size, seq_length, self.heads, d_k).transpose(0, 2, 1, 3)
        self.v = (x@self.parameter[4]).reshape(batch_size, seq_length, self.heads, d_k).transpose(0, 2, 1, 3)
        att_logits = self.q@self.k.transpose(0,1,3,2)  /np.sqrt(d_k)
        self.attention = self.softmax(att_logits,axis=3)
        value = (self.attention@self.v).transpose(0, 2, 1, 3).reshape(batch_size, seq_length, self.dim)
        o = value @ self.parameter[5]
        return o

    def embedding(self, x):
        return np.array(self.parameter[6][x])

    def relu(self, x):
        return np.maximum(x, 0)


    def forward(self, x):
        self.input = x
        self.x = self.embedding(x)
        self.h0 = self.multihead_attention(self.x)
        self.h1 = self.relu(self.h0.sum(axis=1) @ self.parameter[0])
        self.out = self.softmax(self.h1@ self.parameter[1],axis=1)
        return self.out
    import numpy as np

    def MLP_backward(self, dout):
        dW2 =  self.h1.T@dout
        drelu = (self.h1 > 0).astype(float)
        dh1 = (dout@self.parameter[1].T)*drelu
        dW1 = self.h0.sum(axis=1).T @ (dh1)
        dh0 = dh1 @ self.parameter[0].T 
        return dh0,dW1,dW2
       
    def multihead_attention_backward(self, dh0):
        batch_size, seq_length, _ = self.x.shape
        d_k = int(self.dim / self.heads)

        dvalue = (dh0 @ self.parameter[5].T).reshape(batch_size, seq_length, self.heads, d_k).transpose(0, 2, 1, 3)
        dattention = dvalue @ self.v.transpose(0, 1, 3, 2)
        datt_logits = dattention * (self.attention) * (1 - self.attention)
        dq = (datt_logits @ self.k) / np.sqrt(d_k)
        dk = (datt_logits.transpose(0, 1, 3, 2) @ self.q) / np.sqrt(d_k)
        dv = self.attention @ dvalue
        dquery, dkey, dvalue = [i.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, self.dim) for i in (dq, dk, dv)]
        dx = (dquery @ self.parameter[2].T) + (dkey @ self.parameter[3].T) + (dvalue @ self.parameter[4].T)
        dattention_q = dquery.transpose(0,2,1) @ self.x
        dattention_k = dkey.transpose(0,2,1) @ self.x
        dattention_v = dvalue.transpose(0,2,1) @ self.x
        doproj = dh0.transpose(0,2,1) @ self.h0
        d_embedding_layer = np.zeros_like(self.parameter[6])
        np.add.at(d_embedding_layer, self.input[0], dx[0])
        return d_embedding_layer, dattention_q, dattention_k, dattention_v, doproj




class TLoss():
    def __init__(self, model):
        self.model = model
        self.grad = None
        self.input = None
        self.label = None
        self.output = None

    def __call__(self, label, x):
        batch = label.shape[0]
        out = self.model.forward(x)
        self.input = x
        self.label = label
        epslion = 0.00001
        log_likelihood = -np.log(out[range(batch), label]+epslion)
        self.output = np.mean(log_likelihood)
        return np.mean(log_likelihood)

    # def backward(self):
    #     dx = 0.00001
    #     t = []
    #     for i in self.model.parameter:
    #         grad = np.zeros_like(i)
    #         for j in range(i.size):
    #             i.reshape(-1)[j]+=dx
    #             out = self.model.forward(self.input)
    #             batch = self.label.size
    #             epslion = 0.00001
    #             log_likelihood = -np.log(out[range(batch), self.label]+epslion)
    #             dy=log_likelihood.mean()-self.output
    #             dy_dx = dy/dx
    #             grad.reshape(-1)[j]=dy_dx
    #         t.append(grad)
    #     return t
    def backward(self):
        batch_size = self.label.shape[0]
        dout = self.model.out.copy()
        dout[np.arange(batch_size), self.label] -= 1
        dout /= batch_size
        dh0,dW1, dW2 = self.model.MLP_backward(dout)
        dh0=np.expand_dims(dh0, axis=1)
        ones = np.ones((batch_size,500,1))
        dh0 = ones@dh0
        dx, dattention_q, dattention_k, dattention_v, doproj = self.model.multihead_attention_backward(dh0)

        gradients = [dW1, dW2, dattention_q.sum(axis=0), dattention_k.sum(axis=0), dattention_v.sum(axis=0), doproj.sum(axis=0),dx]
        return gradients

class Optimizer():
    def __init__(self,params,beta,lr) :
        self.params = params
        self.beta = beta
        self.lr = lr
        self.moment = [np.zeros_like(i) for i in self.params]
        self.s = [np.zeros_like(i) for i in self.params]
        self.step = 1
        self.epsilon = 0.0000000001
    def __call__(self,grad):
        for i,_ in enumerate(grad):
            self.moment[i] = self.beta[0]*self.moment[i] + (1-self.beta[0])*grad[i]
            self.s[i] = self.beta[1]*self.s[i] + (1-self.beta[1])*grad[i]*grad[i]
            grad[i] = self.lr * (self.moment[i]/(1-self.beta[0]**self.step))/(np.sqrt(self.s[i]/(1-self.beta[1]**self.step))+self.epsilon)
            self.params[i] -= grad[i]



if __name__ == "__main__":
    text,label,vocalbulary_size = get_data("train")
    # print(text)
    model = Transformer(64,2,label.size,vocalbulary_size)
    loss = TLoss(model)
    lr = 0.01
    Adam = Optimizer(model.parameter,(0.9,0.99),lr)
    for i in range(100):
        print( loss(label,text))
        Adam(loss.backward())

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
        self.embedding_layer = np.random.randn(vocabulary_size, dim)
        # self.attention_qkv = [attention_q,attention_k,attention_v]
        w1 = np.random.randn(dim, dim*2)
        w2 = np.random.randn(dim*2, 2)
        # self.parameter = [w1,  w2, attention_q,attention_k,attention_v,oproj,embedding_layer]
        self.parameter = [w1,  w2, attention_q,attention_k,attention_v,oproj]
    def softmax(self, x,axis):
        x_exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return x_exp / np.sum(x_exp, axis=axis, keepdims=True)

    def multihead_attention(self,x):
        batch_size,seq_length,_ = x.shape
        d_k = int(self.dim/self.heads)
        q = (x@self.parameter[2]).reshape(batch_size, seq_length, self.heads, d_k).transpose(0, 2, 1, 3)
        k = (x@self.parameter[3]).reshape(batch_size, seq_length, self.heads, d_k).transpose(0, 2, 1, 3)
        v = (x@self.parameter[4]).reshape(batch_size, seq_length, self.heads, d_k).transpose(0, 2, 1, 3)
        # print(q.shape,k.transpose(0,1,3,2).shape)
        att_logits = q@k.transpose(0,1,3,2)  /np.sqrt(d_k)
        attention = self.softmax(att_logits,axis=3)
        value = (attention@v).transpose(0, 2, 1, 3).reshape(batch_size, seq_length, self.dim)
        o = value @ self.parameter[5]
        return o

    def embedding(self, x):
        # return np.array(self.parameter[6][x])
        return np.array(self.embedding_layer[x])

    def relu(self, x):
        return np.maximum(x, 0)

    def MLP(self,x):
        x = x @ self.parameter[0] 
        x = self.relu(x)
        x = x @ self.parameter[1]
        return self.softmax(x,axis=1)

    def forward(self, x):
        self.x = self.embedding(x)
        self.h0 = self.multihead_attention(self.x)
        self.out = self.MLP(self.h0.sum(axis= 1))
        return self.out
    import numpy as np

    # def backward(self, label):
    #     batch = label.shape[0]
    #     d_out = self.out.copy()
    #     d_out[range(batch), label] -= 1
    #     d_out /= batch

    #     # Gradients for MLP
    #     dw2 = np.dot(self.h_0.T, d_out)
    #     dh_0 = np.dot(d_out, self.parameter[1].T)
    #     dh_0[self.h_0 <= 0] = 0
    #     dw1 = np.dot(self.x.T, dh_0)

    #     # Gradients for multihead attention
    #     dh0 = np.dot(dh_0, self.parameter[0].T)
    #     dh0 = dh0.reshape(self.h0.shape)
    #     dattention = np.dot(dh0, self.parameter[5].T)
    #     dattention = dattention.reshape(self.attention.shape)

    #     # Gradients for attention_q, attention_k, attention_v, and oproj
    #     d_attention_q = np.einsum('bhij,bhik->bijk', dattention, self.attention).sum(axis=0)
    #     d_attention_k = np.einsum('bhij,bhik->bijk', dattention.transpose(0, 1, 3, 2), self.attention).sum(axis=0)
    #     d_attention_v = np.einsum('bhij,bhik->bijk', self.attention, dattention).sum(axis=0)
    #     d_oproj = np.einsum('bhij,bhik->bijk', dattention, self.value).sum(axis=0)

    #     # Gradients for embedding_layer
    #     d_x = np.dot(dh0.reshape(self.x.shape), self.parameter[2].T)
    #     d_embedding_layer = np.zeros_like(self.parameter[6])
    #     np.add.at(d_embedding_layer, self.input, d_x)

    #     grads = [dw1, dw2, d_attention_q, d_attention_k, d_attention_v, d_oproj, d_embedding_layer]
    #     return grads

# ... (Loss and Optimizer classes remain the same)

class Loss():
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

    def backward(self):
        dx = 0.00001
        t = []
        for i in self.model.parameter:
            grad = np.zeros_like(i)
            for j in range(i.size):
                i.reshape(-1)[j]+=dx
                out = self.model.forward(self.input)
                batch = self.label.size
                epslion = 0.00001
                log_likelihood = -np.log(out[range(batch), self.label]+epslion)
                dy=log_likelihood.mean()-self.output
                dy_dx = dy/dx
                grad.reshape(-1)[j]=dy_dx
            t.append(grad)
        return t

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
    model = Transformer(2,2,label.size,vocalbulary_size)
    loss = Loss(model)
    lr = 0.01
    Adam = Optimizer(model.parameter,(0.9,0.99),lr)
    for i in range(10):
        print( loss(label,text))
        Adam(loss.backward())
# class Transformer():
#     # ...

#     def MLP_backward(self, dout):
#         drelu2 = dout @ self.parameter[2].T
#         dW2 = self.relu(self.h1 @ self.parameter[1]) @ drelu2
#         drelu1 = drelu2 @ self.parameter[1].T * (self.h1 > 0)
#         dW1 = self.h0 @ drelu1
#         return dW1, dW2

#     def multihead_attention_backward(self, dout):
#         batch_size, seq_length, _ = self.x.shape
#         d_k = int(self.dim / self.heads)
#         dvalue = (dout @ self.parameter[6].T).reshape(batch_size, self.heads, seq_length, d_k)
#         dattention = dvalue.transpose(0, 2, 1, 3) @ self.v
#         datt_logits = dattention * self.softmax(self.att_logits) * (1 - self.softmax(self.att_logits))
#         dq = (datt_logits @ self.k) / np.sqrt(d_k)
#         dk = (datt_logits.transpose(0, 1, 3, 2) @ self.q) / np.sqrt(d_k)
#         dv = self.attention @ dvalue
#         dquery, dkey, dvalue = [i.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, self.dim) for i in (dq, dk, dv)]
#         dx = (dquery @ self.parameter[3].T) + (dkey @ self.parameter[4].T) + (dvalue @ self.parameter[5].T)
#         return dx, dquery, dkey, dvalue

#     def forward(self, x):
#         self.x = self.embedding(x)
#         self.h0 = self.multihead_attention(self.x)
#         self.h1 = self.relu(self.x @ self.parameter[0])
#         self.out = self.MLP(self.h1)
#         return self.out


# class Loss():
#     # ...

#     def backward(self):
#         batch_size = self.label.shape[0]
#         dout = self.model.out.copy()
#         dout[np.arange(batch_size), self.label] -= 1
#         dout /= batch_size
#         dW1, dW2 = self.model.MLP_backward(dout)
#         dx, dquery, dkey, dvalue = self.model.multihead_attention_backward(dx)
#         dW3 = self.model.x.T @ dquery
#         dW4 = self.model.x.T @ dkey
#         dW5 = self.model.x.T @ dvalue
#         gradients = [dW1, dW2, dW3, dW4, dW5]
#         return gradients


import numpy as np
class model():
    def __init__(self) :
        self.param = [0,1,2] 
class test():
    def __init__(self,data) :
        self.data= data
    def __call__(self) :
        self.data[2] +=1
m = model()
T=test(m.param)
T()
print(T.data)
print(m.param)

m = np.array([[0,2,3],[1,6,5],[9,5,5]])

embedding_layer = np.random.randn(12, 5)
embedding_layer[m]
print(embedding_layer[m])
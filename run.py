import numpy as np
from dataset import get_train_data,get_test_data
from perceptron import perceptron,loss
import tqdm 
if __name__ == "__main__":
    bs = 10
    lr =0.01
    model = perceptron(28*28,10)
    Loss = loss(model)
    training_data,training_target = get_train_data()
    test_data,test_target = get_test_data()
    for epoch in range(100): # train 100 epochs
        train_loss = 0
        for i in range(training_data.shape[0] // bs):   # 每次取一个minibatch的数据
            data = training_data[i*bs : (i+1)*bs]
            target = training_target[i*bs : (i+1)*bs]
            train_loss += Loss(target,data)
            grad=Loss.backward()
            model.parameter[0] -= lr*grad[0]
            model.parameter[1] -= lr*grad[1]
            model.parameter[2] -= lr*grad[2]
            model.parameter[3] -= lr*grad[3]
        train_loss = train_loss /bs
        print(f'epoch {epoch}, training loss: {train_loss}')
        
        if (epoch+1) % 10 == 0:
            correct = 0
            for i in range(test_data.shape[0] // bs):
                data = test_data[i*bs : (i+1)*bs]
                target = test_target[i*bs : (i+1)*bs]
                test_loss = Loss(target,data) # 使用交叉熵
                out=perceptron.forward(data) 
                pred = out.argmax(dim=1, keepdim=True)   # 预测的标签
                correct += pred.eq(target.view_as(pred)).sum().item()
            print(f'epoch {epoch}, test loss: {loss}, Accuracy: {correct / test_data.shape[0]}')
      

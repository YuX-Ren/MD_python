import numpy as np
from minist_dataset import get_train_data,get_test_data
from IMDb_dataset import get_data,get_voca
from transformer import Transformer,Optimizer,TLoss
from perceptron import Perceptron,Loss
import tqdm 
import sys
import pickle
if __name__ == "__main__":
    if sys.argv[1] == 'minist':
        bs = 100
        lr =0.01
        model = Perceptron(28*28,10)
        loss = Loss(model)
        training_data,training_target = get_train_data()
        test_data,test_target = get_test_data()
        for epoch in range(100): # train 100 epochs
            train_loss = 0
            iteration = training_data.shape[0] // bs
            for i in range(iteration):   # 每次取一个minibatch的数据
                data = training_data[i*bs : (i+1)*bs]
                target = training_target[i*bs : (i+1)*bs]
                train_loss += loss(target,data)
                grad=loss.backward()
                model.parameter[0] -= lr*grad[0]
                model.parameter[1] -= lr*grad[1]
                model.parameter[2] -= lr*grad[2]
                model.parameter[3] -= lr*grad[3]
            train_loss = train_loss /iteration
            print(f'epoch {epoch}, training loss: {train_loss}')
            
            if (epoch+1) % 10 == 0:
                correct = 0
                for i in range(test_data.shape[0] // bs):
                    data = test_data[i*bs : (i+1)*bs]
                    target = test_target[i*bs : (i+1)*bs]
                    test_loss = loss(target,data) # 使用交叉熵
                    out=model.forward(data) 
                    pred = out.argmax( axis =1)   # 预测的标签
                    correct += (pred==target.reshape(pred.shape)).sum().item()
                print(f'epoch {epoch}, test loss: {test_loss}, Accuracy: {correct / test_data.shape[0]}')
    elif sys.argv[1] == 'transformer':
        voca,vocalbulary_size =get_voca()
        with open('test.pkl','xb') as f:
            pickle.dump(voca,f)
        # with open('test.pkl','rb') as f:
        #     voca=pickle.load(f)
        # text,label = get_data("train",voca)
        # model = Transformer(64,2,label.size,len(voca))
        # loss = TLoss(model)
        # lr = 0.01
        # Adam = Optimizer(model.parameter,(0.9,0.99),lr)
        # for i in range(100):
        #     print( loss(label,text))
        #     Adam(loss.backward())
        # bs = 10
        # lr =0.0001
        # model = Transformer(64,2,500,len(voca))
        # loss = TLoss(model)
        # Adam = Optimizer(model.parameter,(0.9,0.99),lr)
        # training_data,training_target = get_data("train",voca)
        # test_data,test_target = get_data("test",voca)
        # for epoch in range(100): # train 100 epochs
        #     train_loss = 0
        #     iteration = training_data.shape[0] // bs
        #     for i in range(iteration):   # 每次取一个minibatch的数据
        #         data = training_data[i*bs : (i+1)*bs]
        #         target = training_target[i*bs : (i+1)*bs]
        #         train_loss += loss(target,data)
        #         grad=loss.backward()
        #         Adam(grad)
        #     train_loss = train_loss /iteration
        #     print(f'epoch {epoch}, training loss: {train_loss}')
            
        #     if (epoch+1) % 10 == 0:
        #         correct = 0
        #         for i in range(test_data.shape[0] // bs):
        #             data = test_data[i*bs : (i+1)*bs]
        #             target = test_target[i*bs : (i+1)*bs]
        #             test_loss = loss(target,data) # 使用交叉熵
        #             out=model.forward(data) 
        #             pred = out.argmax( axis =1)   # 预测的标签
        #             correct += (pred==target.reshape(pred.shape)).sum().item()
        #         print(f'epoch {epoch}, test loss: {test_loss}, Accuracy: {correct / test_data.shape[0]}')


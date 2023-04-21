import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def norm(img):
    img = img/255.0
    return img.reshape(-1,28*28)

def get_train_data():
    f = open('./data/train-images-idx3-ubyte', 'rb')
    xs = np.array(np.frombuffer(f.read(), np.uint8, offset=16))
    f.close()
    f = open('./data/train-labels-idx1-ubyte', 'rb')
    ys = np.array(np.frombuffer(f.read(), np.uint8, offset=8))
    f.close()
    xs = np.reshape(xs, (-1, 28, 28, 1)).astype(np.float64)
    xs = norm(xs)
    ys = ys.astype(np.int8)
    return xs,ys

def get_test_data():
    f = open('./data/t10k-images-idx3-ubyte', 'rb')
    xs = np.array(np.frombuffer(f.read(), np.uint8, offset=16))
    f.close()
    f = open('./data/t10k-labels-idx1-ubyte', 'rb')
    ys = np.array(np.frombuffer(f.read(), np.uint8, offset=8))
    f.close()
    xs = np.reshape(xs, (-1, 28, 28, 1)).astype(np.float64)
    xs = norm(xs)
    ys = ys.astype(np.int8)
    return xs,ys


if __name__ == "__main__":
    xs,ys = get_train_data()
    plt.figure()
    for t in range(10):
        plt.subplot(1,10,1+t)
        plt.imshow(xs[t])
        plt.axis("off")
        plt.title(f"t={t}")
    print(norm(xs)[0])
    print(ys[0:10])
    plt.show()
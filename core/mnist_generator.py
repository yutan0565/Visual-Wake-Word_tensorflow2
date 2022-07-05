import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
import numpy as np

mnist = tf.keras.datasets.cifar10
print(type(mnist))

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


print(x_train.shape)
print(y_train.shape)


class DataLoader(Sequence):
	def __init__(self, data_list, batch_size, *args, **kwargs):
		super(DataLoader, self).__init__(*args, **kwargs)
		self.data_list = np.array_split(data_list, len(data_list) // batch_size)
        
	def __getitem__(self, index):
		return self.data_list[index]
        
	def __len__(self):
		return len(self.data_list)




class DataGenerator(Sequence):
    def __init__(self, X, y, batch_size, dim, n_channels, n_classes, shuffle = True):
        self.X = X
        self.y = y if y is not None else y
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __len__(self):
        return int(np.floor(len(self.X) / self.batch_size))
    
    def __data_generation(self, X_list, y_list):
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype = int)
        
        if y is not None:
            # 지금 같은 경우는 MNIST를 로드해서 사용하기 때문에
            # 배열에 그냥 넣어주면 되는 식이지만,
            # custom image data를 사용하는 경우 
            # 이 부분에서 이미지를 batch_size만큼 불러오게 하면 됩니다. 
            for i, (img, label) in enumerate(zip(X_list, y_list)):
                X[i] = img
                y[i] = label
                
            return X, to_categorical(y, num_classes = self.n_classes)
        
        else:
            for i, img in enumerate(X_list):
                X[i] = img
                
            return X
        
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        X_list = [self.X[k] for k in indexes]
        
        if self.y is not None:
            y_list = [self.y[k] for k in indexes]
            X, y = self.__data_generation(X_list, y_list)
            return X, y
        else:
            y_list = None
            X = self.__data_generation(X_list, y_list)
            return X



# dg = DataGenerator(x_train, y_train, 4, (32, 32, 3), 3, 10)

# print(dg)
# print("dfddddddddddddddddddddd")

# import matplotlib.pyplot as plt
                    
# for i, (x, y) in enumerate(dg):

#     if(i <= 1):
#         print(type(i), i)
#         print(type(x),x.shape)
#         print(type(y), y)
#         x_first = x[0]
#         plt.title(y[0])
#         plt.imshow(x_first)



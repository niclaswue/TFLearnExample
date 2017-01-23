import numpy as np 
import tflearn

from tflearn.data_utils import load_csv

#load data from e_commerce.csv, target column is user_action (last), 4 different output classes
data, labels = load_csv('e_commerce.csv', target_column=-1, categorical_labels=True, n_classes=4) 

data = np.array(data, dtype=np.float32) #convert data to numpy float array

#neural network
net = tflearn.input_data(shape=[None, 5]) # 5 inputs, None for batch processing
net = tflearn.fully_connected(net, 32)	# 32 hidden layers
net = tflearn.fully_connected(net, 32)	# another 32 hidden layers
net = tflearn.fully_connected(net, 4, activation='softmax') # 4 outputs, activation method is softmax
net = tflearn.regression(net) # we are using regression to minimize the error

model = tflearn.DNN(net) #create DNN model

model.fit(data, labels, n_epoch=20, batch_size=16, show_metric=True) #training 20 epochs

print("Training completed! ")
print("Testing:")

#some data from the end of e_commerce.csv to test the DNN
test1 =  np.array([1,3,0.731593700135,0,0], dtype=np.float32)
test2 = np.array([0,0,6.36877488837,1,3], dtype=np.float32)
test3 = np.array([0,0,0.172853398207,1,3], dtype=np.float32)
test4 = np.array([1,0,0.20996439824,0,3], dtype=np.float32)
test5 = np.array([0,0,2.61688195401,1,3], dtype=np.float32)

pred = model.predict([test1, test2, test3, test4, test5]) #make predictions

print("test1 prediction (2 is correct) ", pred[0])
print("test2 prediction (0 is correct) ", pred[1])
print("test3 prediction (0 is correct) ", pred[2])
print("test4 prediction (0 is correct) ", pred[3])
print("test5 prediction (0 is correct) ", pred[4])
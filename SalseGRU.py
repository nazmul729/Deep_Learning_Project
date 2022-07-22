import numpy as np 
import pandas as pd  #to read CSV file
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, GRU
from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

df = pd.read_csv("sales-of-shampoo.csv")
df_remove_Feature = df.drop(['Date'], 1, inplace=True)

#Split into training and testing
number_of_prediction_samples = 5
df_train= df[:len(df) - number_of_prediction_samples]
df_test= df[len(df) - number_of_prediction_samples:]

##print("Original Training: ",df_train)
#Normalize and prepare for training
training_data = df_train.values
training_data = min_max_scaler.fit_transform(training_data)

x_training_data = training_data[0:len(training_data)-1]
y_training_data = training_data[1:len(training_data)]
x_training_data = np.reshape(x_training_data, (len(x_training_data), 1, 1))
print(x_training_data)

#Train the model
number_of_units = 10
activation_function = 'relu'
optimizer_name = 'adam'
loss_function = 'mean_squared_error'
batch_size = 5
num_epochs = 2000

# Initialize the RNN

#build model
model = Sequential()
model.add(GRU(units=number_of_units, input_shape=(None,1), return_sequences=False))
#model.add(NearestNeighbors(n_neighbors=2, algorithm='ball_tree'))
model.add(Activation('tanh'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('relu'))
#neigh = NearestNeighbors(n_neighbors=1)
model.compile(loss='mse', optimizer='adam')
model.summary()
model.fit(x_training_data, y_training_data, batch_size = batch_size, epochs = num_epochs,verbose=0)


#Prediction here
test_set = df_test.values

inputs = np.reshape(test_set, (len(test_set), 1))
inputs = min_max_scaler.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))

predicted_price = model.predict(inputs)
predicted_price = min_max_scaler.inverse_transform(predicted_price)
#print(regressor.score(x_test, y_test))
print("Done2222222222222222")
#Visualize results
plt.figure(figsize=(10, 10), dpi=40, facecolor = 'w', edgecolor = 'k')
plt.plot(test_set[:, 0], color='red', label='Real BTC Price')
plt.plot(predicted_price[:, 0], color = 'blue', label = 'Predicted BTC Price')

plt.title('BTC Price Prediction', fontsize = 40)
plt.xlabel('Time', fontsize=40)
plt.ylabel('BTC Price(USD)', fontsize = 40)
plt.legend(loc = 'best')
plt.show()

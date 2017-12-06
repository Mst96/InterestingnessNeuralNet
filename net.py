import keras
import csv
import os
import numpy as np
import textToMatPairs as txt2mat
import h5py
#Linear model
# from keras.models import Sequential
#Dropout - technique that switches off nodes at random
#Dense - fully connected layer
#Flatten - Reshapes the matrix to feed into other layers
# from keras.layers import Dropout, Dense, Flatten, Embedding

# from keras.layers import Conv2D, MaxPooling2D
# from keras import backend as K

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.losses import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Flatten, MaxPooling1D
from keras.layers import Conv1D, GlobalMaxPooling1D



#gets rid of links and other things to clean the tweet up so it's just text.
def load_train_data(num_tweets):
	with open("TESTING AND TRAINING DATA/training_trump4.csv", 'r') as f:
		tweets = list(csv.reader(f, delimiter=","))
	# print(tweets)
	train_array = np.asarray(tweets)[1:num_tweets]
	train_array = train_array[:, [2,9]]
	# return array[:, [2,9]]
	
	return train_array

def load_test_data(num_tweets):
	with open("TESTING AND TRAINING DATA/testing_trump2.csv", 'r') as f:
		tweets = list(csv.reader(f, delimiter=","))
	
	test_array = np.asarray(tweets)[1:num_tweets]
	test_array = test_array[:, [2,9]]

	return test_array

def pad(tweets, vocab_size):
	# integer encode the documents
	encoded = [one_hot(word, vocab_size) for word in tweets]
	print(encoded)
	max_length = 50
	padded_docs = pad_sequences(encoded, maxlen=max_length, padding='post')
	return padded_docs

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_403_trained_model.h5'
# set parameters:
max_features = 501
maxlen = 676
vocab_size = 50
#32 samples in a batch, all processed independently
batch_size = 100
embedding_dims = 100
filters = 100
kernel_size = 3
hidden_dims = 250
#Arbitrary cutoff point. 1 = 1 PASS OVER DATASET.
epochs = 2


#initialise input dimensions
#split data into training and testing



# tweet = Embedding(output_dim=512, input_dim=676, input_length=10)(main_input)
# mentions = Input(shape=(10,), name='hashtags+media+@')
# hashtags = Input(shape=(10,), name='hashtags+media+@')
# media = Input(shape=(10,), name='hashtags+media+@')

# x = Input(shape=(26, 26, 4), name='hashtags+media+@')

print('Loading data...')
train = load_train_data(max_features)
x_train = train[:,0]
y_train = train[:,1]
test = load_test_data(max_features)
x_test = test[:,0]
y_test = test[:,1]

train_array = []
test_array = []
print(x_train.shape)
for x in x_train:
	train_array.append(txt2mat.textToMatPairs(x))


for x in x_test:
	test_array.append(txt2mat.textToMatPairs(x))

x_train = np.asarray(train_array)
# x_train = np.transpose(train_array)
x_test = np.asarray(test_array)
# x_test = np.transpose(test_array)
# x_train = pad(x_train, vocab_size)
# x_test = pad(x_test, vocab_size)
print(x_train.shape)
#build model
print('Build model...')
model = Sequential()


# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(vocab_size,
                    embedding_dims,
                    input_length=maxlen))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(32))
model.add(Conv1D(32, (3,), padding='same'))
model.add(Activation('relu'))
# model.add(Conv1D(32, (3,)))
# model.add(Activation('relu'))
# model.add(MaxPooling1D(pool_size=(2,)))
model.add(Dropout(0.25))
# model.add(Conv1D(64, (3,), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv1D(64, (3,)))
# model.add(Activation('relu'))
# model.add(MaxPooling1D(pool_size=(2,)))
# model.add(Dropout(0.25))
model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('relu'))
# print(model.output_shape)
# model.add(Dense(1, activation='sigmoid'))

# # we use max pooling:
# # model.add(GlobalMaxPooling1D())


# # We add a vanilla hidden layer:
# model.add(Dense(hidden_dims))
# model.add(Dropout(0.2))
# model.add(Activation('relu'))


# # We project onto a single unit output layer, and squash it with a sigmoid:
# model.add(Dense(1))
# model.add(Activation('sigmoid'))

# x = keras.layers.concatenate([x, auxiliary_input])


# model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, retweets, likes])

#train model
model.compile(loss='mean_squared_error',
              optimizer='adam', metrics=['mean_squared_error'])

print(model.summary())

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
#Print results
score = model.evaluate(x_test, y_test, batch_size=batch_size)
# x = x_test[50:90]
# print(x.shape)
# print(x_train.shape)
prediction = model.predict(x_test)
print(y_test)
print(prediction)
# for i, y in enumerate(prediction):
# 	print(i)
# 	print(prediction[i][0])
# 	print(y_train[50+i])
	# sub = prediction[i][0] - y_train[50+i]
	# print(np.mean(np.square(sub)))
# # error = mean_squared_error(y_test, prediction)
# print(y_test)
# print(prediction)

#Save model
#save structure of model as json
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

import keras
import csv
import os
import numpy as np
# import textToMatPairs as txt2mat
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
from keras.models import Model
from keras.layers import Dense, Dropout, Activation, concatenate, Reshape
from keras.layers import Embedding, Flatten, MaxPooling1D, LSTM
from keras.layers import Input,Conv1D, GlobalMaxPooling1D, BatchNormalization



#gets rid of links and other things to clean the tweet up so it's just text.
def load_train_data():
	with open("TESTING AND TRAINING DATA/training.csv", 'r') as f:
		tweets = list(csv.reader(f, delimiter=","))
	# print(tweets)
	train_array = np.asarray(tweets)[1:]
	train_array = train_array[:, [0, 5, 6, 7, 8, 9, 10 ]]
	# return array[:, [2,9]]
	
	return train_array

def load_test_data():
	with open("TESTING AND TRAINING DATA/testing.csv", 'r') as f:
		tweets = list(csv.reader(f, delimiter=","))
	
	test_array = np.asarray(tweets)[1:]
	test_array = test_array[:, [0, 5, 6, 7, 8, 9, 10 ]]
	return test_array

def pad(tweets, vocab_size):
	# integer encode the documents
	encoded = [one_hot(word, vocab_size) for word in tweets]
	max_length = 100
	padded_docs = pad_sequences(encoded, maxlen=max_length, padding='post')
	return padded_docs

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_403_trained_model.h5'
# set parameters:
max_features = 501
maxlen = 676
vocab_size = 100
#32 samples in a batch, all processed independently
batch_size = 100
embedding_dims = 100
filters = 100
kernel_size = 3
hidden_dims = 250
#Arbitrary cutoff point. 1 = 1 PASS OVER DATASET.
epochs = 100


#initialise input dimensions
#split data into training and testing



# tweet = Embedding(output_dim=512, input_dim=676, input_length=10)(main_input)
# mentions = Input(shape=(10,), name='hashtags+media+@')
# hashtags = Input(shape=(10,), name='hashtags+media+@')
# media = Input(shape=(10,), name='hashtags+media+@')

# x = Input(shape=(26, 26, 4), name='hashtags+media+@')

print('Loading data...')
train = load_train_data()

#main input
tweets_train = train[:,0]

#outputs
retweets_train = train[:,1]
likes_train = train[:,2]
score_train = train[:,3]

# print(retweets_train)
#other inputs
links_train = train[:,4]
# links_train = links_train.astype(np.int32)
mentions_train = train[:,5]
hashtags_train = train[:,6]
# mentions_train = mentions_train.astype(np.int32)
# hashtags_train = hashtags_train.astype(np.int32)

test = load_test_data()
#main input
tweets_test = test[:,0]

#outputs
retweets_test = test[:,1]
print(retweets_test)
likes_test = test[:,2]
print(likes_test)
score_test = test[:,3]

#other inputs
links_test = test[:,4]
mentions_test = test[:,5]
hashtags_test = test[:,6]


# links_train = links_train.astype(np.int32)
# mentions_train = mentions_train.astype(np.int32)
# hashtags_train = hashtags_train.astype(np.int32)

# train_array = []
# test_array = []
# for x in tweets_train:
# 	train_array.append(txt2mat.textToMatPairs(x))


# for x in tweets_test:
# 	test_array.append(txt2mat.textToMatPairs(x))

# tweets_train = np.asarray(train_array)
# x_train = np.transpose(train_array)
# tweets_test = np.asarray(test_array)
# x_test = np.transpose(test_array)
tweets_train = pad(tweets_train, vocab_size)
tweets_test = pad(tweets_test, vocab_size)
#build model
print('Build model...')
# model = Sequential()


# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
main_input = Input(shape=(100,), name='main_input')
# embedded = Embedding(vocab_size,
#                     embedding_dims,
#                     input_length=maxlen)(main_input)

# flatten_out = Flatten()(embedded)
# x = concatenate([flatten_out, mentions_dense, hashtags_dense, links_dense])
# x = Reshape(500)(x)
# print(x)
x = Dense(80, activation='relu')(main_input)
x = Dense(50, activation='relu')(x)
x = Dense(10, activation='relu')(x)
x = Dense(5, activation='relu')(x)

mentions = Input(shape=(1,), name='mentions_input')
mentions_dense = Dense(1, )(mentions)
hashtags = Input(shape=(1,), name='hashtags_input')
hashtags_dense = Dense(1, )(hashtags)
links = Input(shape=(1,), name='links_input')
links_dense = Dense(1, )(links)

x = concatenate([x, mentions_dense, hashtags_dense, links_dense])
# x = Dropout(0.2)(x)
# model.add(Flatten())
# model.add(Dense(32))
# x = Conv1D(64, (2,), activation='relu')(x)
# x = MaxPooling1D(pool_size=(2,))(x)
# x = Dropout(0.2)(x)
# x = Flatten()(x)
# model.add(Conv1D(32, (3,)))
# model.add(Activation('relu'))
# model.add(MaxPooling1D(pool_size=(2,)))
# model.add(Dropout(0.25))
# model.add(Conv1D(64, (3,), padding='same'))
# model.add(Activation('relu'))
# model.add(Conv1D(64, (3,)))
# model.add(Activation('relu'))
# model.add(MaxPooling1D(pool_size=(2,)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
score = Dense(1, activation='sigmoid', name='score_output')(x)
retweets = Dense(1, activation='sigmoid', name='retweets_output')(x)
likes = Dense(1, activation='sigmoid', name='likes_output')(x)
# print(model.output_shape)
# model.add(Dense(1, activation='sigmoid'))
model = Model(inputs=[main_input, mentions, hashtags, links], outputs=[score, retweets, likes])


#train model
model.compile(loss='mean_squared_error',
              optimizer='adam')


x_train = [tweets_train, mentions_train, hashtags_train, links_train] 
y_train = [score_train, retweets_train, likes_train]
x_test = [tweets_test, mentions_test, hashtags_test, links_test]
y_test = [score_test, retweets_test, likes_test]


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
#Print results
score = model.evaluate(x_test, y_test, batch_size=batch_size)
# x = x_test[50:90]
# print(x.shape)
# print(x_train.shape)
prediction = model.predict(x_train, batch_size)
print(type(prediction))
print(y_test)
print(prediction)

pred = np.swapaxes(prediction,0,1)
np.savetxt('output.csv', pred, delimiter=",")

# for i, y in enumerate(prediction):
# 	print(prediction[0])
	# print(y_train[i])
	# sub = prediction[0][i] - y_train[i]
	# print(np.mean(np.square(sub)))
# error = mean_squared_error(y_test, prediction)
# print(y_test)
# print(prediction)

#Save model
#save structure of model as json
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

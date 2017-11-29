import pandas as pd
import numpy as np

PATH = '/Users/mustafa/documents/Year 4/403/'

trump_tweets_df = pd.read_csv(PATH + 'clean_trump.csv')

trump_tweets_df['split'] = np.random.randn(trump_tweets_df.shape[0], 1)

#Split tweets randomly in ratio 7:3
msk = np.random.rand(len(trump_tweets_df)) <= 0.7

train = trump_tweets_df[msk]
test = trump_tweets_df[~msk]

#Run 5 times and change filename to get 5 different sets of data for cross validation
train.to_csv(PATH + 'training_trump5.csv')
test.to_csv(PATH + 'testing_trump5.csv')
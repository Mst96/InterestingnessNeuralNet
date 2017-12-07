import pandas as pd
import numpy as np

PATH = '/Users/mustafa/documents/Year 4/403/'
FILE = 'new_trump.csv'
trump_tweets_df = pd.read_csv(PATH + FILE)

#checks if there is an @ in the tweet
def bool_to_int(bool):
    if bool is True:
        return 1
    return 0


#clean up tweets and find out if there are mentions or media
trump_tweets_df['media'] = trump_tweets_df['media'].map(bool_to_int)
trump_tweets_df['mention'] = trump_tweets_df['mention'].map(bool_to_int)
trump_tweets_df['hashtag'] = trump_tweets_df['hashtag'].map(bool_to_int)

msk = np.random.rand(len(trump_tweets_df)) < 0.7
train = trump_tweets_df[msk]
test = trump_tweets_df[~msk]

print(len(train))

print(len(test))

train.to_csv(PATH + "training.csv")
test.to_csv(PATH + "testing.csv")

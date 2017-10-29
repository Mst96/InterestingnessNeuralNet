import pandas as pd

trump_tweets_df = pd.read_csv('/Users/mustafa/documents/Year 4/403/trump.csv', nrows=150)

#gets rid of links and other things to clean the tweet up so it's just text.
def clean_tweet(tweet):
    links_removed = []
    parts = tweet.lower().split()
    for part in parts:
        if 'https' in part:
            print(part)
        elif '&amp;' in part:
            print(part)
        else:
            links_removed.append(part)
    print(links_removed)
    return ' '.join(links_removed)

#checks if there is an image link in the tweet
def if_media(tweet):
    parts = tweet.split()
    for part in parts:
        if 'https' in part:
            return True
    return False

#checks if there is an @ in the tweet
def if_mention(tweet):
    parts = tweet.split()
    for part in parts:
        if '@' in part:
            return True
    return False

#change - to / in times for consistency
def clean_times(time):
    new_time = list(time)
    for x, char in enumerate(new_time):
        if char == '-':
            new_time[x] = '/'
    return ''.join(new_time)

#Drop source and is_retweet columns as they are irrelevant
#is_retweet is all false
trump_tweets_df = trump_tweets_df.drop(['is_retweet','source'], axis=1)

#clean up tweets and find out if there are mentions or media
trump_tweets_df['media'] = trump_tweets_df['text'].map(if_media)
trump_tweets_df['mention'] = trump_tweets_df['text'].map(if_mention)
trump_tweets_df['text'] = trump_tweets_df['text'].map(clean_tweet)

trump_tweets_df['created_at'] = trump_tweets_df['created_at'].map(clean_times)
print(trump_tweets_df['created_at'])

#length may be important
trump_tweets_df['length'] = trump_tweets_df['text'].map(lambda x: len(x))
print(trump_tweets_df['length'])

# Parse datetime
trump_tweets_df['created_at'] = pd.to_datetime(trump_tweets_df['created_at'], infer_datetime_format=True)
trump_tweets_df['time'] = trump_tweets_df['created_at'].dt.time
trump_tweets_df['date'] = trump_tweets_df['created_at'].dt.date

trump_tweets_df.to_csv('/Users/mustafa/documents/Year 4/403/new_trump.csv')

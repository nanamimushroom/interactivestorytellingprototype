import tweepy
import pandas as pd  


consumer_key = "9401zvA875KB5DxhakbgSBSXP"
consumer_secret = "NhiVu1rgojrtW3tOrsiLir2ixxB49l8rtQTOYOFB30oNgv6IMA"
access_token = "723447412430245888-tf8HQbB2cg6VYdr5qsDjLzcxojbXHkC"
access_token_secret = "hAtuRvjsukBSR4XI8JkhZRRzLYxQg3iBfu26FLcttLq3p"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True)

#search according to keywords
search_words = 'as a woman' 
#search_words = "feminist" "sexist"
date_since = "2020-5-1"
new_search = search_words + " -filter:retweets"

#find the data I want  
tweets = tweepy.Cursor(api.search,q=new_search,lang = 'en',since=date_since,tweet_mode='extended').items(200) 
#select the column I need
latest_tweets = [[tweet.full_text] for tweet in tweets]

tweets_data = pd.DataFrame(latest_tweets)
tweets_data.columns = ['text']
  
#print(tweets_data.info())

# saving the dataframe 
tweets_data.to_csv('difficultwoman0503.csv')



 
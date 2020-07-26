---
title: "Twitter (Manchester United) - Sentiment Analysis - Python"
date: 2020-07-21
tags: [Python, Twitter, Data Science, Sentiment Analysis]
header:
  image: 
excerpt: "Python, Twitter, Data Science, Sentiment Analysis"
mathjax: "true"
---

![alt]({{ site.url }}{{ site.baseurl }}/images/TwitAna/manuni.png)


# Social Media and Network Analytics Manchester United Twitter Account

<div style="text-align: justify"> Nowadays, soccer is one of the most trending topics in newspapers, social media, Sports TV and radio, and even more if it is related to the English tournament, which is the most powerful one.
This project will focus its attention in Manchester United FC Twitter account, as it is the biggest club in England. The Accountancy firm Deloitte estimates that Manchester United has 75 million fans worldwide, while other estimates put this figure closer to 333 million . The club also has 71
million fans in social media, which makes it the third football club with most social media followers in the world after Real Madrid and Barcelona.</div>

<div style="text-align: justify"> This project will consist in extract Data from Twitter, clean it, visualize it and finally get some insights.</div>

<div style="text-align: justify"> Thankfully Twitter provides their own API that allows users to collect information from a particular account Time Line or topic, among others.
The data collection technique used for this analysis was REST, which allows the users to query a twitter account only by registering in the API site. Note that when registering, the API will generate a set of Keys(passwords) that would allow you to pull the data required.
The data collected was stored in a .json file , that allows the user to query any single feature in the Twitter corpus in order to analyse the downloaded dataset.</div>

- Note: For this excersice we are only extracting the last 1000 tweets.

After pulling the data from the twitter API, I import the needed packages.

```python
import tweepy           # To consume Twitter's API
import pandas as pd
from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from collections import Counter
import json
import tweepy
tweets = []
```

The first step executed was to open tha .json file where the information was stored:


```python
if __name__ == '__main__':
    fname = 'ManUtd'
    with open(fname, 'r') as f:
        for line in f:
            tweet = json.loads(line)
            tweets.append(tweet)
print("Number of tweets extracted: {}.\n".format(len(tweets)))
```

    Number of tweets extracted: 1000.


The second step executed was to get some basic statistics and printed information of the information collected as follow:

- The first 5 Tweets:

```python
# We print the most recent 5 tweets:
for tweet in tweets[:5]:
    print(tweet['text'])
    print()
```

    "Watching him I thought, he's definitely got something." 👀

    We caught up with former #MUFC scout Jim Ryan in our latest UTD Podcast 🎧

    📖 Our #MUNWHU issue of #UnitedReview is now available to purchase.

    Click below to order your copy and any other ed… https://t.co/dt6qsDUrpV

    We’re glad you’re okay, @EricBailly24 ❤️

    #MUFC https://t.co/O9gxpRwFZL

    The latest fitness update for our penultimate #PL match of the season on Wednesday 👇

    #MUFC #MUNWHU

    ⚽️ A classic #GoalOfTheDay, brought to you by Yorkie and @VanCole9.

    #MUFC https://t.co/rZxKicpSq0


In order to organize the data, I create a dataframe using Pandas and display again the first Ten tweets:

```python
#Creating a (pandas) DataFrame

# create a pandas dataframe as follows:
data = pd.DataFrame(data=[tweet['text'] for tweet in tweets], columns=['Tweets'])

# display the first 10 elements of the dataframe:
display(data.head(10))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>"Watching him I thought, he's definitely got s...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>📖 Our #MUNWHU issue of #UnitedReview is now av...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>We’re glad you’re okay, @EricBailly24 ❤️\n\n#M...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>The latest fitness update for our penultimate ...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>⚽️ A classic #GoalOfTheDay, brought to you by ...</td>
    </tr>
    <tr>
      <td>5</td>
      <td>All eyes on the run-in 👊\n\n#MUFC #PL</td>
    </tr>
    <tr>
      <td>6</td>
      <td>RT @MarcusRashford: There’s no where to hide t...</td>
    </tr>
    <tr>
      <td>7</td>
      <td>It's time to dust ourselves off and prepare fo...</td>
    </tr>
    <tr>
      <td>8</td>
      <td>"We are really sad about the result and losing...</td>
    </tr>
    <tr>
      <td>9</td>
      <td>💬 "We're very disappointed but we need to be b...</td>
    </tr>
  </tbody>
</table>
</div>

Twitter API provides multiple fields per twitter, then I print the mosre relavants for the analysis.


```python
# print info from the first tweet:
print(tweets[0]['id'])
print(tweets[0]['created_at'])
print(tweets[0]['source'])
print(tweets[0]['favorite_count'])
print(tweets[0]['retweet_count'])
print(tweets[0]['geo'])
print(tweets[0]['coordinates'])
print(tweets[0]['entities'])
```

    1285290880925085696
    Mon Jul 20 19:10:00 +0000 2020
    <a href="https://ads-api.twitter.com" rel="nofollow">Twitter for Advertisers</a>
    2812
    199
    None
    None
    {'hashtags': [{'text': 'MUFC', 'indices': [85, 90]}], 'symbols': [], 'user_mentions': [], 'urls': []}


<div style="text-align: justify"> After checking which information was pulled from the relevant fields, I continue to include the relevant fields to the previously created dataframe and print again the first Ten tweets:</div>


```python
# Adding relevant info to our dataframe

# We add relevant data:
data['len']  = np.array([len(tweet['text']) for tweet in tweets])
data['ID']   = np.array([tweet['id'] for tweet in tweets])
data['Date'] = np.array([tweet['created_at'] for tweet in tweets])
data['Source'] = np.array([tweet['source'] for tweet in tweets])
data['Likes']  = np.array([tweet['favorite_count'] for tweet in tweets])
data['RTs']    = np.array([tweet['retweet_count'] for tweet in tweets])

data['Date']=pd.to_datetime(data.Date)
# Display of first 10 elements from dataframe:
display(data.head(10))
#print(data.dtypes)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweets</th>
      <th>len</th>
      <th>ID</th>
      <th>Date</th>
      <th>Source</th>
      <th>Likes</th>
      <th>RTs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>"Watching him I thought, he's definitely got s...</td>
      <td>133</td>
      <td>1285290880925085696</td>
      <td>2020-07-20 19:10:00+00:00</td>
      <td>&lt;a href="https://ads-api.twitter.com" rel="nof...</td>
      <td>2812</td>
      <td>199</td>
    </tr>
    <tr>
      <td>1</td>
      <td>📖 Our #MUNWHU issue of #UnitedReview is now av...</td>
      <td>140</td>
      <td>1285270496653848582</td>
      <td>2020-07-20 17:49:00+00:00</td>
      <td>&lt;a href="https://ads-api.twitter.com" rel="nof...</td>
      <td>1858</td>
      <td>112</td>
    </tr>
    <tr>
      <td>2</td>
      <td>We’re glad you’re okay, @EricBailly24 ❤️\n\n#M...</td>
      <td>71</td>
      <td>1285255401542496259</td>
      <td>2020-07-20 16:49:01+00:00</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>18045</td>
      <td>1108</td>
    </tr>
    <tr>
      <td>3</td>
      <td>The latest fitness update for our penultimate ...</td>
      <td>99</td>
      <td>1285244172941885440</td>
      <td>2020-07-20 16:04:24+00:00</td>
      <td>&lt;a href="https://ads-api.twitter.com" rel="nof...</td>
      <td>3437</td>
      <td>241</td>
    </tr>
    <tr>
      <td>4</td>
      <td>⚽️ A classic #GoalOfTheDay, brought to you by ...</td>
      <td>99</td>
      <td>1285200032476454914</td>
      <td>2020-07-20 13:09:00+00:00</td>
      <td>&lt;a href="https://studio.twitter.com" rel="nofo...</td>
      <td>4262</td>
      <td>409</td>
    </tr>
    <tr>
      <td>5</td>
      <td>All eyes on the run-in 👊\n\n#MUFC #PL</td>
      <td>35</td>
      <td>1285176628335673344</td>
      <td>2020-07-20 11:36:00+00:00</td>
      <td>&lt;a href="https://ads-api.twitter.com" rel="nof...</td>
      <td>9992</td>
      <td>672</td>
    </tr>
    <tr>
      <td>6</td>
      <td>RT @MarcusRashford: There’s no where to hide t...</td>
      <td>140</td>
      <td>1285166311308500993</td>
      <td>2020-07-20 10:55:00+00:00</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>0</td>
      <td>6926</td>
    </tr>
    <tr>
      <td>7</td>
      <td>It's time to dust ourselves off and prepare fo...</td>
      <td>97</td>
      <td>1285150203952209920</td>
      <td>2020-07-20 09:51:00+00:00</td>
      <td>&lt;a href="https://ads-api.twitter.com" rel="nof...</td>
      <td>7156</td>
      <td>490</td>
    </tr>
    <tr>
      <td>8</td>
      <td>"We are really sad about the result and losing...</td>
      <td>140</td>
      <td>1285125803165048833</td>
      <td>2020-07-20 08:14:02+00:00</td>
      <td>&lt;a href="http://www.falcon.io" rel="nofollow"&gt;...</td>
      <td>13992</td>
      <td>1067</td>
    </tr>
    <tr>
      <td>9</td>
      <td>💬 "We're very disappointed but we need to be b...</td>
      <td>102</td>
      <td>1284958943689576448</td>
      <td>2020-07-19 21:11:00+00:00</td>
      <td>&lt;a href="https://ads-api.twitter.com" rel="nof...</td>
      <td>7510</td>
      <td>604</td>
    </tr>
  </tbody>
</table>
</div>

<div style="text-align: justify"> After creating the full dataframe, I will continue to extract interesting information available as follows using Numpy package:</div>

- The Tweet with more likes and its word length:

```python
# extract the tweet with more FAVs and more RTs:

fav_max = np.max(data['Likes'])
rt_max  = np.max(data['RTs'])

fav = data[data.Likes == fav_max].index[0]
rt  = data[data.RTs == rt_max].index[0]

# Max FAVs:
print("The tweet with more likes is: \n{}".format(data['Tweets'][fav]))
print("Number of likes: {}".format(fav_max))
print("{} characters.\n".format(data['len'][fav]))
```

    The tweet with more likes is:
    A hero. An inspiration. One of our own.

    We are so proud of you, @MarcusRashford ❤️ https://t.co/haAb0m2I4u
    Number of likes: 197739
    107 characters.



- The Tweet with more retweets and its word length:


```python
# Max RTs:
print("The tweet with more retweets is: \n{}".format(data['Tweets'][rt]))
print("Number of retweets: {}".format(rt_max))
print("{} characters.\n".format(data['len'][rt]))
```

    The tweet with more retweets is:
    A hero. An inspiration. One of our own.

    We are so proud of you, @MarcusRashford ❤️ https://t.co/haAb0m2I4u
    Number of retweets: 33131
    107 characters.

<div style="text-align: justify"> Having the Length of the Twitters, how many likes and retweets they can have, allowed me to retrieve the average Twitter’s length:</div>

- Average Length of twitters in Manchester United Account:


```python
#Visualization and basic statistics

#Averages and popularity

# extract the mean of lengths:
mean = np.mean(data['len'])

print("The lenght's average in tweets: {}".format(mean))
```

    The length's average in tweets: 92.554


<div style="text-align: justify"> Other useful information is the user action source, which let us know how users are posting their tweets, as in the following pie Chart.</div>


```python
# obtain all possible sources:
sources = []
for source in data['Source']:
    if source not in sources:
        sources.append(source)

# print sources list:
print("Creation of content sources:")
for source in sources:
    print("* {}".format(source))
```

    Creation of content sources:
    * <a href="https://ads-api.twitter.com" rel="nofollow">Twitter for Advertisers</a>
    * <a href="http://twitter.com/download/iphone" rel="nofollow">Twitter for iPhone</a>
    * <a href="https://studio.twitter.com" rel="nofollow">Twitter Media Studio</a>
    * <a href="https://mobile.twitter.com" rel="nofollow">Twitter Web App</a>
    * <a href="http://www.falcon.io" rel="nofollow">Falcon Social Media Management </a>
    * <a href="https://ads.twitter.com" rel="nofollow">Twitter Ads</a>
    * <a href="http://live.playingsurface.net/" rel="nofollow">Playing Surface</a>
    * <a href="https://www.twitter.com/" rel="nofollow">Twitter Media Studio - LiveCut</a>



```python
#create a numpy vector mapped to labels:
percent = np.zeros(len(sources))

for source in data['Source']:
    for index in range(len(sources)):
        if source == sources[index]:
            percent[index] += 1
            pass


percent /= 100

# Pie chart:
pie_chart = pd.Series(percent, index=sources, name='Sources')
pie_chart.plot.pie(fontsize=11, autopct='%.2f', figsize=(6, 6));
plt.title('Source Precentages - Pie Chart')
plt.show()
```


![alt]({{ site.url }}{{ site.baseurl }}/images/TwitAna/TA1.png)


<div style="text-align: justify"> It is also important to check the twitter’s length through the time, which is visualized on the following graph.</div>


```python
# Time series

#  time series for data:

tlen = pd.Series(data=data['len'].values, index=data['Date'])
tfav = pd.Series(data=data['Likes'].values, index=data['Date'])
tret = pd.Series(data=data['RTs'].values, index=data['Date'])


# Lenghts along time:
tlen.plot(figsize=(16,4), color='r');
plt.ylabel('Count')
plt.title('Twitters Lenght -Time Series')
plt.show()
```


![alt]({{ site.url }}{{ site.baseurl }}/images/TwitAna/TA2.png)


<div style="text-align: justify"> It is also interesting how the Tweets are Liked or Retweeted  through the time, which can be visualized in the following graph.</div>


```python
# Likes vs retweets visualization:
tfav.plot(figsize=(16,4), label="Likes", legend=True)
tret.plot(figsize=(16,4), label="Retweets", legend=True);
plt.ylabel('Count')
plt.title('Retweets vs Likes - Time Series')
plt.show()
```


![alt]({{ site.url }}{{ site.baseurl }}/images/TwitAna/TA3.png)


<div style="text-align: justify"> This source of plots is extremely helpful to understand the user’s behaviour, as we could note when the Manchester United account was more active and could match with particular facts happening in these days.</div>

<div style="text-align: justify"> For example, the chart elucidates that there was an event that made the Manchester United followers be more active around the 17th of June of this year, in terms of Likes, which match with the re-start of the Premier League after the COVID-19 shutdown.</div>

### Sentiment Analysis

<div style="text-align: justify"> In order to identify what are the perceptions and feelings towards Manchester United, the use of unsupervised sentiment analysis on a set of Tweets was performed, using the Textblob approach.</div>

<div style="text-align: justify"> This approach is basically the same as word counting but uses an algorithm that has an integrated dictionary of positive and negative words and only classifies the tweets as Positive 1, negative 0 or neutral 0, as follow:</div>


```python
# Sentiment analysis

#Importing textblob

from textblob import TextBlob
import re

def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def analize_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1

#create a column with the result of the analysis:
data['SA'] = np.array([ analize_sentiment(tweet) for tweet in data['Tweets'] ])
plt.plot(data['SA']) # plotting by columns
plt.ylabel('Sentiment -> 1 Positive, 0 Neutral, -1 NEgative')
plt.xlabel('Twetts')
plt.title('Sentiment Analysis - TextBlod Method')
plt.show()
```


![alt]({{ site.url }}{{ site.baseurl }}/images/TwitAna/TA4.png)

<div style="text-align: justify"> Once we have the sentiment per tweet, I print the full data frame with the new SA field to see the results of this analysis in our full dataframe.</div>

```python
# display the updated dataframe with the new column:
display(data.head(10))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweets</th>
      <th>len</th>
      <th>ID</th>
      <th>Date</th>
      <th>Source</th>
      <th>Likes</th>
      <th>RTs</th>
      <th>SA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>"Watching him I thought, he's definitely got s...</td>
      <td>133</td>
      <td>1285290880925085696</td>
      <td>2020-07-20 19:10:00+00:00</td>
      <td>&lt;a href="https://ads-api.twitter.com" rel="nof...</td>
      <td>2812</td>
      <td>199</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>📖 Our #MUNWHU issue of #UnitedReview is now av...</td>
      <td>140</td>
      <td>1285270496653848582</td>
      <td>2020-07-20 17:49:00+00:00</td>
      <td>&lt;a href="https://ads-api.twitter.com" rel="nof...</td>
      <td>1858</td>
      <td>112</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>We’re glad you’re okay, @EricBailly24 ❤️\n\n#M...</td>
      <td>71</td>
      <td>1285255401542496259</td>
      <td>2020-07-20 16:49:01+00:00</td>
      <td>&lt;a href="http://twitter.com/download/iphone" r...</td>
      <td>18045</td>
      <td>1108</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>The latest fitness update for our penultimate ...</td>
      <td>99</td>
      <td>1285244172941885440</td>
      <td>2020-07-20 16:04:24+00:00</td>
      <td>&lt;a href="https://ads-api.twitter.com" rel="nof...</td>
      <td>3437</td>
      <td>241</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>⚽️ A classic #GoalOfTheDay, brought to you by ...</td>
      <td>99</td>
      <td>1285200032476454914</td>
      <td>2020-07-20 13:09:00+00:00</td>
      <td>&lt;a href="https://studio.twitter.com" rel="nofo...</td>
      <td>4262</td>
      <td>409</td>
      <td>1</td>
    </tr>
    <tr>
      <td>5</td>
      <td>All eyes on the run-in 👊\n\n#MUFC #PL</td>
      <td>35</td>
      <td>1285176628335673344</td>
      <td>2020-07-20 11:36:00+00:00</td>
      <td>&lt;a href="https://ads-api.twitter.com" rel="nof...</td>
      <td>9992</td>
      <td>672</td>
      <td>0</td>
    </tr>
    <tr>
      <td>6</td>
      <td>RT @MarcusRashford: There’s no where to hide t...</td>
      <td>140</td>
      <td>1285166311308500993</td>
      <td>2020-07-20 10:55:00+00:00</td>
      <td>&lt;a href="https://mobile.twitter.com" rel="nofo...</td>
      <td>0</td>
      <td>6926</td>
      <td>1</td>
    </tr>
    <tr>
      <td>7</td>
      <td>It's time to dust ourselves off and prepare fo...</td>
      <td>97</td>
      <td>1285150203952209920</td>
      <td>2020-07-20 09:51:00+00:00</td>
      <td>&lt;a href="https://ads-api.twitter.com" rel="nof...</td>
      <td>7156</td>
      <td>490</td>
      <td>-1</td>
    </tr>
    <tr>
      <td>8</td>
      <td>"We are really sad about the result and losing...</td>
      <td>140</td>
      <td>1285125803165048833</td>
      <td>2020-07-20 08:14:02+00:00</td>
      <td>&lt;a href="http://www.falcon.io" rel="nofollow"&gt;...</td>
      <td>13992</td>
      <td>1067</td>
      <td>1</td>
    </tr>
    <tr>
      <td>9</td>
      <td>💬 "We're very disappointed but we need to be b...</td>
      <td>102</td>
      <td>1284958943689576448</td>
      <td>2020-07-19 21:11:00+00:00</td>
      <td>&lt;a href="https://ads-api.twitter.com" rel="nof...</td>
      <td>7510</td>
      <td>604</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
</div>

And Finally I print the percentage of positives, negatives and neutral sentiment:

```python
#Analyzing the results
# construct lists with classified tweets:

pos_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] > 0]
neu_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] == 0]
neg_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] < 0]


#print percentages:

print("Percentage of positive tweets: {}%".format(len(pos_tweets)*100/len(data['Tweets'])))
print("Percentage of neutral tweets: {}%".format(len(neu_tweets)*100/len(data['Tweets'])))
print("Percentage de negative tweets: {}%".format(len(neg_tweets)*100/len(data['Tweets'])))
```

    Percentage of positive tweets: 36.9%
    Percentage of neutral tweets: 55.0%
    Percentage de negative tweets: 8.1%


<div style="text-align: justify"> After obtaining all the Statistics, plots, tables and Models applied over the 1000 tweets in the Official Manchester United Twitter account, we can conclude that Manchester United FC is a highly active account.</div>
  
<div style="text-align: justify"> Twitter Manchester United’s fans are highly active, liking and retweeting the official Twitter account, the most liked tweet has 197739 and the most retweeted one has 33131.</div>
  
<div style="text-align: justify"> Sentiment Analysis allows us to conclude that despite the team has lost the FA Cup Semifinal against Chelsea FC and is not yet qualified to the 2020/2021 Champions League, the fans are more inclined to provided positive opinions towards the team though the Twitter account.</div>
  
<div style="text-align: justify"> Twitter API gives the chance to analyse the unstructured data and apply sentiment algorithms to understand people feelings or opinions towards a specific user or account.</div>


---
title: "Sentiment Analysis - Twitter - Manchester United"
date: 2020-07-21
tags: [Python, Twitter, Data Science, Sentiment Analysis]
header:
  image: 
excerpt: "Python, Twitter, Data Science, Sentiment Analysis"
mathjax: "true"
---

![alt]({{ site.url }}{{ site.baseurl }}/images/TwitAna/manuni.png)


# Social Media and Network Analytics Manchester United Twitter Account

Nowadays, soccer is one of the most trending topics in newspapers, social media, Sports TV and radio, and even more if it is related to the English tournament, which is the most powerful one.
This report will focus its attention in Manchester United FC Twitter account, as it is the biggest club in England. There are three top reasons to consider a soccer club big, which are:

- Fans
- Earnings
- Tournaments

The Accountancy firm Deloitte estimates that Manchester United has 75 million fans worldwide, while other estimates put this figure closer to 333 million . The club also has 71
million fans in social media, which makes it the third highest social media following in the world after Real Madrid and Barcelona.
The Club was the Highest earning football club in the world in 2016-2017, while it has 20 Premier League titles, 3 champions leagues, 12 FA cups, 5 League Cups and 21 FA Community Shields.
These three reasons confirm that Manchester United is an interesting Topic to be analysed, as their fans are probably actively posting in their social media.
Since Sr. Alex Fergusson (most winner coach in the team history) left the team in 2013, the club has not been performing as it used to be, they have already had three different top-level coaches.
This situation creates a controversial feeling within the fans, as they are supporting the biggest team, while in the other hand the team is not achieving any fan expectations in the pitch.
Applying Sentiment and topic analysis to Manchester United’s Twitter account, would probably elucidates how fans are feeling with the team and what are the most relevant topics in its account.

## Data Collection
Thankfully Twitter provides their own API that allows users to collect information from a particular account Time Line or topic, among others.
The data collection technique used for this analysis was REST, which allows the users to query a twitter account without any permission through OAuth authentication, being that the main reason for using it.
The data collected was stored in a .jason file, that allows the user to query any single feature in the Twitter corpus in order to analyse the downloaded dataset.


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

The first step executed was to get some basic statistics and printed information of the information collected as follow:


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



The first ten Twitters:


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


Relevant Information to analyse:


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


When gathering the relevant information, it was created a table with the 10 most recent Tweets matched with the available information available.


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


Having the Length of the Twitters, how many likes and retweets they can have, allowed to retrieve the average Twitter’s length, the most Liked and retweeted one.
Average Length of twitters in Manchester United Account:


```python
#Visualization and basic statistics

#Averages and popularity

# extract the mean of lenghts:
mean = np.mean(data['len'])

print("The lenght's average in tweets: {}".format(mean))
```

    The lenght's average in tweets: 92.554


The more Liked one with its length:


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



The most Retweeted one with its length:


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



Other useful information is the user action source, which let us know how users are posting their tweets, as in the following pie Chart.


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


It is also important to check the twitter’s length through the time, which is visualized on the following graph.


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


Tweets may be Liked or Retweeted, which can be visualized in the following graph through the time.


```python
# Likes vs retweets visualization:
tfav.plot(figsize=(16,4), label="Likes", legend=True)
tret.plot(figsize=(16,4), label="Retweets", legend=True);
plt.ylabel('Count')
plt.title('Retweets vs Likes - Time Series')
plt.show()
```


![alt]({{ site.url }}{{ site.baseurl }}/images/TwitAna/TA3.png)


This source of plots is extremely helpful to understand the user’s behaviour, as we could note when the Manchester United account was more active and could match with particular facts happening in these days.
For example, the chart elucidates that there was an event that made the Manchester United followers be more active around the 17th of June of this year, in terms of Likes, which match with the re-start of the Premier League after the COVID-19 shutdown.

## Analysis Approach

In order to identify what are the perceptions and feelings towards Manchester United, the use of unsupervised sentiment analysis on a set of Tweets was performed, using the Textblob approach.


### Sentiment Analysis

### Textblob Approach

This approach is basically the same as word counting but uses an algorithm that has an integrated dictionary of positive and negative words and only classifies the tweets as Positive 1, negative 0 or neutral 0, as follow:




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


## Analysis & Insights

After obtaining all the Statistics, plots, tables and Models applied over the 1000 tweets in the Official Manchester United Twitter account, the next step is to analyse them to find insights hidden in the unstructured dataset.

Firstly, it is important to see what information the Twitter API is providing to plan what sort of analysis are possible to do. In this case we can retrieve ID, Source, Date, Tweet, Likes, Retweets Geo, Coordinates, ect.

The first insight gained was the Twitter average length in the last 1000 tweets for ManUtd FC Account, which is 92 words length. We were also able to see the Tweet “A hero. An inspiration. One of our own. We are so proud of you, @MarcusRashford ❤️ https://t.co/haAb0m2I4u” with more Likes (197739), referring to the recent Marcus Rashford donation to the poor guys in Manchester during the pandemic of 2020. Conincidentially the most retweeted post was the same as the omre liked one with 33131 retweets, which confirms how happy the fans are with Marcus donation.

Social Media and Network Analytics Manchester United Twitter Account
Having the source of the tweet, it is also important to note that a 56.20 % of the users post their activity through twitter web client, followed by falcon social media management with 23.20%, snappy tv.com with 9.7%, Twitter ads Composer with 9.1% and Twitter for iPhone with 1.8%.
The variety of topics related to Manchester united were elucidated though the Hashtag frequency and words frequency.
The most frequents Hashtags are mufc, ucl, facup, mutv, to name a few, and they are referring to the team itself, the UEFA Champions League, the FA Cup and Manchester United TV Channel, which let us understand that the Manchester United twitter account has relationship through hashtags with other independent accounts.
The most repeated words were goal, win, @romelolukaku9, game, old, Liverpool, to name a few. This approach also allows us to understand the sort of topics posted by the fans towards the team.

Likes and Retweets visualization is highly important to be analysed. This time series tells where the most controversial topics happened, as we can see that they are pics in the time series, for instance, Manchester United beat Liverpool around 19th March 2018, and we can tell in the plot that the fans were highly active this day liking the Manchester United account or retweeting their tweets.
Using three Sentiment Analysis allows us to see the trend in how the fans are feeling towards the team, despite their approaches are highly similar it is helpful to conclude having. Them all.

The Textblob approach provided the following percentages, which confirms the team has a unbalanced distributions of feelings with 50% positives sentiments, 38% negatives and 11% neutral, which is what we can conclude from the plot.

Percentage of positive tweets: 50.1%
Percentage of neutral tweets: 38.5%
Percentage de negative tweets: 11.4%

Finally, the topic analysis provides some useful information, such as what words are being
used in the different topics where Manchester United Twitter account is linked to, to understand how different topics could be related to each other.
For example, in the most important topic “mufc old trafford ball alexis_sanchez carras16 team football jose”, the most used words are “mufc, facup, marcusrashford, etc”, and for the second topic “mufc ucl sevilla makes change munliv make visitors way d_degea” the same words are highly used, so we can infer that the topics are related to each other.

## Conclusions

Manchester United FC is a highly active account, the club has more than 1000 tweets per month.
Twitter Manchester United’s fans are highly active, liking and retweeting the official Twitter account, the most liked tweet has 197739 and the most retweeted one has 33131.
Sentiment Analysis allows us to conclude that despite the team has lost the FA Cup Semifinal against Chelsea FC and is not yet qualified to the 2020/2021 Champions League, the fans are more inclined to provided positive opinions towards the team though the Twitter account.
Twitter API gives the chance to analyse the unstructured data and apply sentiment algorithms to understand people feelings or opinions towards a specific user or account.


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


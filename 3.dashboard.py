from audioop import avg
from multiprocessing.sharedctypes import Value
from typing import Any
import streamlit as st
import pandas as pd
import streamlit as st
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import requests
from io import BytesIO
from collections import Counter
import numpy as np
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time
from textblob import TextBlob
from decimal import Decimal, ROUND_UP

# import pandas
df = pd.read_csv('charts_data_lyrics_added.csv', sep=';')

# create variable df_1 for structuring purposes and beging able to reuse the original df later on
df_1 = df

# streamlit title
st.title("Spotify song lyrics word occurrence and sentiment")

#---- wordcloud ------
# source: https://stackoverflow.com/a/58554736/16659122

for lyrics in df_1['lyrics']:
        all_lyrics = lyrics
        

url = 'https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_CMYK_Green.png'
response = requests.get(url)
mask = np.array(Image.open(BytesIO(response.content)))
image_colors = ImageColorGenerator(mask)
wordcloud = WordCloud(width=1600, height=800, background_color="rgba(255, 255, 255, 0)", mask=mask
                ,color_func = image_colors).generate_from_text(all_lyrics)


# Display the generated wordcloud
st.subheader("Explore the word count and it's associated sentiment of Dutch top 200 most popular monthly Spotify songs from 2017 to 2022")
st.caption("*List is shortened to the monthy top 20 due to time limit concerns retrieving song lyrics using the Genius API, see documentation.")
st.write("___")
st.set_option('deprecation.showPyplotGlobalUse', False)
plt.figure( figsize=(20,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
st.pyplot()
st.write("___")


#---- User input for word occurence ------
options = st.multiselect(
    "Select a word to check the occurence in song lyrics",
    options=["love", "money", "hate", "bad", "I", "me", "you", "baby"], # convert to list
    default=["love"]
)
input = ' '.join(map(str, options))

# dynamic streamlit title
if input:
        st.write("Number of times", "'",input,"'","occurs in the Spotify *top 20 songs.")

#---- data cleaning and word occurence, grouped by month ------
df_1 = df_1.drop("Unnamed: 0", axis=1)
df_1 = df_1.drop("Unnamed: 0.1", axis=1)
df_1["word_count"] = df_1["lyrics"].str.count(" " + input)
df_1["date"]= pd.to_datetime(df_1["date"], format='%d/%m/%Y')
df_1 = df_1.groupby(pd.Grouper(key='date', freq='MS')).agg({'word_count': ['sum'], 'streams': ['sum']})
df_selection = pd.DataFrame()
df_selection["word count"] = df_1["word_count"]['sum']
df_selection["streams"] = df_1["streams"]['sum']
df_selection = df_selection.reset_index()

print(df_selection)


#---- sentiment analysis ------
# source: https://teams.microsoft.com/l/message/19:656b155d5af74201acec0ada2a44e63b@thread.tacv2/1666610343802?tenantId=98932909-9a5a-4d18-ace4-7236b5b5e11d&groupId=efbaaf1f-5dec-49dd-a9bb-8983634f970d&parentMessageId=1666610343802&teamName=Master%20Data-driven%20Design%202022-2023%20(students)&channelName=A%20-%20Fundamentals%20of%20Data%20Science&createdTime=1666610343802&allowXTenantAccess=false
# an attempt on a sentiment analysis

# create variable df_1 for structuring purposes and beging able to reuse the original df later on
df_2 = df

df_sentiment = df_2[df_2['lyrics'].str.contains(" " + input)]

#drop "Unnamed: 0.1", "Unnamed: 0" and "rank"
df_sentiment = df_sentiment.drop("Unnamed: 0.1", axis=1)
df_sentiment = df_sentiment.drop("Unnamed: 0", axis=1)
df_sentiment = df_sentiment.drop("rank", axis=1)

# In this case the review column from my quickly made reviews file
df_sentiment["sentiment_score"] = df_sentiment["lyrics"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# now we can add a column where we label these -1 to +1 values into a category
df_sentiment["sentiment"] = np.select([df_sentiment["sentiment_score"] < 0, df_sentiment["sentiment_score"] == 0, df_sentiment["sentiment_score"] > 0], ['negative', 'neutral', 'positive'])

# mean of all song lyrics containing the word (user input)
sentiment_score = (df_sentiment["sentiment_score"]).mean()

# again counting the amount of times a word occurs, and make it descending (on the top will be the song where the word occurs the most in)
df_sentiment["word_count"] = df_sentiment["lyrics"].str.count(" " + input)
df_sentiment.loc[(df_sentiment["word_count"]==-1),"word_count"] = 0
df_sentiment = df_sentiment.sort_values(by='word_count', axis=0, ascending=False)

# specific df for the sentiment line chart per month > CHECK
df_sentiment["date"] = pd.to_datetime(df_sentiment["date"], format='%d/%m/%Y')

# error handling when user selects multiple words which do not occur in the song lyrics
try:
        df_sentiment_temp = df_sentiment.groupby(pd.Grouper(key='date', freq='MS')).agg({'sentiment_score': ['mean']})
        df_sentiment_line_chart = pd.DataFrame()
        df_sentiment_line_chart["sentiment_score"] = df_sentiment_temp["sentiment_score"]['mean']
        df_sentiment_line_chart = df_sentiment_line_chart.reset_index()
except:
        st.info("Combination doesn't occur the the dataset, please change your selection", icon="ℹ️")
        st.stop()

print(df_sentiment_line_chart.info())

#------ area chart word occurrence--------
st.area_chart(df_selection, y="word count",x="date")

if input:
        st.write("Sentiment of the song lyrics which contain the word", "'",input,"'.")
        st.caption("*When no monthly average sentiment is displayed in the bar chart, there were no or too few lyrics to measure sentiment.")

#------ bar chart somg lyrics sentiment--------
st.bar_chart(df_sentiment_line_chart, y="sentiment_score",x="date")

st.write("The mean sentiment of the cluster of song lyrics containing the word","'", input,"'", "has a sentiment score of", round(sentiment_score, 2), "on a scale of -1 negative, 0 neutral, +1 positive.")

with st.expander("Raw data"):
        st.caption("*Raw data might contain multiple of the same songs, since a song can appear multiple times in the monthly chart for a few weeks or months.")
        st.write(df_sentiment)

#------ pick the song with the most word occurence based on the selected word by the user -----
song_title = ""
song_artist = ""
song_streams = ""
song_url = ""
song_date = ""
song_text = ""

try:
        song_title = df_sentiment["title"].loc[df_sentiment.index[0]]
        song_artist = df_sentiment["artist"].loc[df_sentiment.index[0]]
        song_streams = df_sentiment["streams"].loc[df_sentiment.index[0]]
        song_url = df_sentiment["url"].loc[df_sentiment.index[0]]
        song_date = df_sentiment["date"].loc[df_sentiment.index[0]]
        song_text = df_sentiment["lyrics"].loc[df_sentiment.index[0]] #.head(1)
        song_word_count = df_sentiment["word_count"].loc[df_sentiment.index[0]]
        song_sentiment = df_sentiment["sentiment"].loc[df_sentiment.index[0]]
        song_sentiment_score = df_sentiment["sentiment_score"].loc[df_sentiment.index[0]]
except IndexError: # error handling when user selects multiple words which do not occur in the song lyrics
        st.info("Combination doesn't occur the the dataset, please change your selection", icon="ℹ️")
        st.stop()



# ---- wordcloud ------
# source: https://stackoverflow.com/a/58554736/16659122


url = 'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.memoriesvideography.com%2Fwp-content%2Fuploads%2F2015%2F12%2FCD.jpg&f=1&nofb=1&ipt=45a8bf510118ddb976cddb6bd9d0556bab4472fd67abba4b423c8364dc9b42cd&ipo=images'
response = requests.get(url)
mask = np.array(Image.open(BytesIO(response.content)))
image_colors = ImageColorGenerator(mask)
wordcloud2 = WordCloud(width=1600, height=800, background_color="rgba(255, 255, 255, 1)", mask=mask
                ,color_func = image_colors).generate_from_text(song_text)      



# Display the generated image:
st.set_option('deprecation.showPyplotGlobalUse', False)
plt.figure( figsize=(20,10))
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis("off")
st.write(" ")
st.subheader("Song in which the selected word occurs the most")

# present the metadate of the song with the most word occurence based on the selected word by the user 
with st.spinner('Loading'):
        time.sleep(1)
        try:
                st.write(song_title, "from",song_artist,)
                st.write("Release date:",song_date)
                st.write("Song lyrics contains", song_word_count, "times the word", "'",input,"'.")
                st.write("The song lyrics of",song_title,"has a sentiment score of", song_sentiment_score.round(2), "which can be considered as" ,song_sentiment,".") 
        except NameError:
                st.info("")

with st.expander("Song lyrics"):
        st.write(song_text)
        st.write("___")
        st.write("Link to",song_title, "from",song_artist,":", song_url)
        st.write("Streams:", song_streams)
    

with st.expander("Wordcloud"):
        with st.spinner('Loading'):
                time.sleep(1)
                st.pyplot()
#!pip install -q pyngrok
#!pip install -q streamlit
import streamlit as st
import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
plt.style.use('fivethirtyeight')
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Dense,Input, Embedding,LSTM,Dropout,Conv1D, MaxPooling1D, GlobalMaxPooling1D,Dropout,Bidirectional,Flatten,BatchNormalization
from tensorflow.python.ops.math_ops import reduce_prod
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import tweepy
from tweepy import OAuthHandler
#To Hide Warnings
st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)
# Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
#sns.set_style('darkgrid')
from nltk.corpus import stopwords
from wordcloud import WordCloud 

STYLE = """
<style>
img {
    max-width: 100%;
}
</style> """

def main():
    """ Common ML Dataset Explorer """
    

    html_temp = """
	<div style="background-color: #00AAEE;"><p style="color:white;font-size:20px;padding:9px">Real Time Sentiment analysis</p></div>
	"""
    st.markdown(html_temp, unsafe_allow_html=True)
    #st.subheader("Select a topic which you'd like to get the real time tweet sentiment analysis on :")

    ################# Twitter API Connection #######################
    
    consumer_key = 'ImIwIsuwkDdFrA4oxX6rhWaCL'
    consumer_secret = 'ueCO37JSmztl0bDcMuQ7wTD7UdH3g33erZk87vbMhnTKVp3zeW'
    access_token = '1591848292161298432-pIgJDR5zxxfFKg4lCL0678IrQ9ICli'
    access_token_secret = 'RH2EuWtV3JmSebLJhIc0QrMzpq0Q6TfQ7xmxLyqSMy8cd'



    # Use the above credentials to authenticate the API.

    auth = tweepy.OAuthHandler( consumer_key , consumer_secret )
    auth.set_access_token( access_token , access_token_secret )
    api = tweepy.API(auth)
    ################################################################
    
    df = pd.DataFrame(columns=["Date","User","IsVerified","Tweet","Likes","RT",'User_location'])
    
    # Write a Function to extract tweets:
    def get_tweets(Topic,Count):
        i=0
        #my_bar = st.progress(100) # To track progress of Extracted tweets
        for tweet in tweepy.Cursor(api.search_tweets, q=Topic,count=Count, lang="en",exclude='retweets').items():
            #time.sleep(0.1)
            #my_bar.progress(i)
            df.loc[i,"Date"] = tweet.created_at
            df.loc[i,"User"] = tweet.user.name
            df.loc[i,"IsVerified"] = tweet.user.verified
            df.loc[i,"Tweet"] = tweet.text
            df.loc[i,"Likes"] = tweet.favorite_count
            df.loc[i,"RT"] = tweet.retweet_count
            df.loc[i,"User_location"] = tweet.user.location
            #df.to_csv("TweetDataset.csv",index=False)
            #df.to_excel('{}.xlsx'.format("TweetDataset"),index=False)   ## Save as Excel
            i=i+1
            if i>Count:
                break
            else:
                pass
    # Function to Clean the Tweet.
    def tweet_cleaner(text):
        tok = WordPunctTokenizer()
        pat1 = r'@[A-Za-z0-9]+'
        pat2 = r'https?://[A-Za-z0-9./]+'
        combined_pat = r'|'.join((pat1, pat2))
        www_pat = r'www.[^ ]+'
        negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                        "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                        "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                        "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                        "mustn't":"must not"}
        neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')
        soup = BeautifulSoup(text, 'lxml')
        souped = soup.get_text()
        try:
            bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")
        except:
            bom_removed = souped
        stripped = re.sub(combined_pat, '', bom_removed)
        stripped = re.sub(www_pat, '', stripped)
        lower_case = stripped.lower()
        neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)
        letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
        clean_repeat = re.sub(r'(.)1+', r'1', letters_only)
        # During the letters_only process two lines above, it has created unnecessay white spaces,
        # I will tokenize and join together to remove unneccessary white spaces
        words = [x for x  in tok.tokenize(clean_repeat) if len(x) > 1]
        return (" ".join(words)).strip()
    
        
    # Funciton to analyze Sentiment
    def analyze_sentiment(tweet):
        text = tweet_cleaner(tweet)

        loaded_CNN_model = load_model('CNN_best_weights.01-0.8340.hdf5')

        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        sequences = tokenizer.texts_to_sequences([text])
        #print(sequences)
        x_seq = pad_sequences(sequences, maxlen=45)
        #print(x_test_seq)
        
        analysis = loaded_CNN_model.predict(x_seq).round()
        if analysis == 0:
            return 'Negative'
        elif analysis == 1:
            return 'Positive'
        else:
            return 'Neutral'
    
    def prepCloud(Topic_text,Topic):
        Topic = str(Topic).lower()
        Topic =' '.join(re.sub('([^0-9A-Za-z \t])', ' ', Topic).split())
        Topic = re.split("\s+",str(Topic))
        stop_words = set(stopwords.words('english'))
        stop_words.update(Topic) 
        text_new = " ".join([txt for txt in Topic_text.split() if txt not in stop_words])
        return text_new

    

    
    ## Collect Input from user :

    Topic = str()
    with st.form(key='my_form'):
        Topic = str(st.text_input("Enter the topic you want to do analysis of tweets on: "))     
        Count = int(st.number_input('Enter the count of tweets(min 10 and max 500) to search for:' ,min_value=10, max_value=500,value=10,step =1)) -1
        submit_button = st.form_submit_button(label='Submit')

    if len(Topic) > 0 :
        
        # Call the function to extract the data. pass the topic and filename you want the data to be stored in.
        with st.spinner("Please wait, Tweets are being extracted"):
            get_tweets(Topic , Count)
        st.success('Tweets have been Extracted !!!!')    
        
    
        # Call function to get Clean tweets
        df['clean_tweet'] = df['Tweet'].apply(lambda x : tweet_cleaner(x))
    
        # Call function to get the Sentiments
        df["Sentiment"] = df["Tweet"].apply(lambda x : analyze_sentiment(x))
    
    # Write Summary of the Tweets
        st.write("Total Tweets Extracted for Topic '{}' are : {}".format(Topic,len(df.Tweet)))
        st.write("Total Positive Tweets are : {}".format(len(df[df["Sentiment"]=="Positive"])))
        st.write("Total Negative Tweets are : {}".format(len(df[df["Sentiment"]=="Negative"])))
        #st.write("Total Neutral Tweets are : {}".format(len(df[df["Sentiment"]=="Neutral"])))

        if st.button("See the Extracted Data"):
            #st.markdown(html_temp, unsafe_allow_html=True)
            st.success("Below is the Extracted Data :")
            st.write(df.head(50))
        
        
        # get the countPlot
        if st.button("Get Count Plot for Different Sentiments"):
            st.success("Generating A Count Plot")
            st.subheader(" Count Plot for Different Sentiments")
            st.write(sns.countplot(df["Sentiment"]))
            st.pyplot()
        
        # Piechart 
        if st.button("Get Pie Chart for Different Sentiments"):
            st.success("Generating A Pie Chart")
            a=len(df[df["Sentiment"]=="Positive"])
            b=len(df[df["Sentiment"]=="Negative"])
            d=np.array([a,b])
            explode = (0.1, 0.1)
            st.write(plt.pie(d,shadow=True,explode=explode,labels=["Positive","Negative"],autopct='%1.2f%%'))
            st.pyplot()
            
            
        # get the countPlot Based on Verified and unverified Users
        if st.button("Get Count Plot Based on Verified and unverified Users"):
            st.success("Generating A Count Plot (Verified and unverified Users)")
            st.subheader(" Count Plot for Different Sentiments for Verified and unverified Users")
            st.write(sns.countplot(df["Sentiment"],hue=df.IsVerified))
            st.pyplot()
        
        
        ## Points to add 1. Make Backgroud Clear for Wordcloud 2. Remove keywords from Wordcloud
        
        
        # Create a Worlcloud
        if st.button("Get WordCloud for all things said about {}".format(Topic)):
            st.success("Generating A WordCloud for all things said about {}".format(Topic))
            text = " ".join(review for review in df.clean_tweet)
            stop_words = set(stopwords.words('english'))
            text_newALL = prepCloud(text,Topic)
            wordcloud = WordCloud(stopwords=stop_words,max_words=800,max_font_size=70).generate(text_newALL)
            st.write(plt.imshow(wordcloud, interpolation='bilinear'))
            st.pyplot()
        
        
        #Wordcloud for Positive tweets only
        if st.button("Get WordCloud for all Positive Tweets about {}".format(Topic)):
            st.success("Generating A WordCloud for all Positive Tweets about {}".format(Topic))
            text_positive = " ".join(review for review in df[df["Sentiment"]=="Positive"].clean_tweet)
            stop_words = set(stopwords.words('english'))
            text_new_positive = prepCloud(text_positive,Topic)
            #text_positive=" ".join([word for word in text_positive.split() if word not in stopwords])
            wordcloud = WordCloud(stopwords=stop_words,max_words=800,max_font_size=70).generate(text_new_positive)
            st.write(plt.imshow(wordcloud, interpolation='bilinear'))
            st.pyplot()
        
        
        #Wordcloud for Negative tweets only       
        if st.button("Get WordCloud for all Negative Tweets about {}".format(Topic)):
            st.success("Generating A WordCloud for all Positive Tweets about {}".format(Topic))
            text_negative = " ".join(review for review in df[df["Sentiment"]=="Negative"].clean_tweet)
            stop_words = set(stopwords.words('english'))
            text_new_negative = prepCloud(text_negative,Topic)
            #text_negative=" ".join([word for word in text_negative.split() if word not in stopwords])
            wordcloud = WordCloud(stopwords=stop_words,max_words=800,max_font_size=70).generate(text_new_negative)
            st.write(plt.imshow(wordcloud, interpolation='bilinear'))
            st.pyplot()
    
    st.sidebar.header("About the App")
    st.sidebar.info("This Real Time Twitter Sentiment analysis Project scraps twitter for the topic selected by the user. The extracted tweets aree used to determine the Sentiments of those tweets.")
    st.sidebar.text("Built with Streamlit")
    
    #st.sidebar.header("For Any Queries/Suggestions Please reach out at :")

    if st.button("Exit"):
        st.balloons()



if __name__ == '__main__':
    main()
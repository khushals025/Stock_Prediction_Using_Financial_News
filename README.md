# Stock Price Prediction using News Articles

<div align="center">
  <img src="https://www.robomarkets.com/blog/wp-content/uploads/2020/10/16-1.jpg" alt= "Image alt" width="1000">
</div>

## Table of Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [News Scraping](#news-scraping)
- [Sentiment Analysis](#sentiment-analysis)
- [LSTM model](#lstm-model)
- [Predictions with News](#predictions-with-news)
- [Accuracy](#accuracy)
- [Future Scope](#future-scope)

## 1. Introduction
This repository showcases stock price prediction using time series analysis and sentiment analysis. Leverage Hugging Face's Roberta model for sentiment analysis on financial news to gauge market sentiment. LSTM-based predictive model captures stock market patterns, while visualizations highlight the sentiment-stock price prediction for S&P 500


## 2. Dependencies

- LSTM model

```bash
import pandas as pd
import numpy as np
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential
from scipy.special import softmax
import tqdm
```

- Wordcloud and visualization 

  
```bash
!pip install wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
```

- Roberta by HuggingFace library

  
```bash
import transformers
print(transformers.__version__) #--> 4.31.0
```
  
```bash
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

```

- VADER( Valence Aware Dictionary for Sentiment Reasoning) module by nltk(natural language toolkit)

  
```bash
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

```

## 3. Dataset
### 3.1 News Articles 
- <a href="https://www.kaggle.com/datasets/jeet2016/us-financial-news-articles">Dataset</a> : US Financial News Articles, which contains 2018 news articles in json format from publications like Bloomberg, CNBC, reuters, wsj(wall street journal) and fortune. The main Zip file contains 5 other folders , each for every month in 2018 from January to May.

#### These news articles could cover a wide range of subjects within the realm of finance, including but not limited to:
- Stock Market News
- Economic Indicators
- Business News
- Financial Regulations
- Investment Insights
- Banking and Finance Industry
- Macroeconomic Trends







### 3.2 Stock data
- Data.zip contains 455 csv files for each company in S&P500 list, with the following attributes:
- Date, Open_x,	High_x,	Low_x,	Close_x,	Adj, Close_x,	Volume_x,	Open_y,	High_y,	Low_y,	Close_y,	Adj Close_y,	Volume_y

- This data ranges from 2007 to 2018. In this project we will be using just the data precisely from 2017-12-07 to 2018-06-01 as this ranges coresponds to the json data's publication dates.
- Which means if we need to use news data for stock price prediction we should be able to use these dates only.

- out of these we will use Date and Adj_close_y which is the closing price of the stock for that corresponding date


## 4. News Scraping 
-  In News Articles dataset A JSON object is enclosed within curly braces {} and consists of key-value pairs. Each key is a string, and the associated value can be of various types: strings, numbers, booleans, arrays, or nested JSON objects.
- For this project we will be using keys such as 'published’(date), 'thread', 'title', 'text', 'url' which are attributes of each article written about S&P500.
- Firstly, we find the paths for all the json files and then store them in a list.
- Then we extract the data using the path into an array of list “json_list”


```bash
for item in file_paths:
    if item.endswith('.json'):
        with open(item) as f:
            for line in f:
                data = json.loads(line)
                json_list.append([data['published'], data['thread']['site'], data['title'], data['text'], data['url']])
```
- convert json data to csv file.

```bash
col_names =  ['published_date','source_name','title','body','url']

df= pd.DataFrame(json_list,columns=col_names)
```

#### Data Pre-Processing for News Articles
- Changed Date-Time formatting into (YYY-MM-DD)
- Dropped 2 NAN values form the data frame.
- Arranged in ascending order of publication date.
- saved it as "US_financial_news_articles_2018.csv"


## 5. Sentiment Analysis
- Used news articles data frame's "title" column to compute tf-idf vectorization to refelct import words from the corpus of articles to just visualize the results.
- Term Frequency (TF): It represents the number of occurrences of a word in a document. It reflects the importance of the word in a specific document.
- l2 norm is used to normalize Term frequency.

<div align="center">
  <img src="https://t1.daumcdn.net/cfile/tistory/99E6814A5DFD4DA70F" alt="Image Alt Text" width ="300">
</div>

- Inverse Document Frequency (IDF): It measures the rarity of a word across all documents in the corpus. It penalizes words that appear frequently in multiple documents and boosts the importance of rare words.
- #### Here, d refers to a document, N is the total number of documents, df is the number of documents with term t.

  <div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:816/1*1pTLnoOPJKKcKIcRi3q0WA.jpeg" alt="Image Alt Text" width = "600">
</div>


- Also used CountVectorizer : converts a collection of text documents to a matrix of token counts. It counts the frequency of each word (token) in each document and represents the results as a sparse matrix. 



- Removing stop words for noise reduction as these words are of little importance compared to other tokens. 
```bash

my_stop_words = list(text.ENGLISH_STOP_WORDS.union(["ap1", "00", "000", "0", "561", "190", "09", "24", "2017","2018", "000 00", "2018",
                               "jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec",
                               "ag", "ap3", "000 00 ap3", "ap2", "00 ap2", "000 00 ap2", "10", "00 ap1",
                               "000 00 ap1", "oct 2018", "000 000", "000 000 00", "october 2018", "10 2018",
                               "11 2018", "november 2018", "12 2018", "december 2018",'11', '12', 'december', 'november', 'october']))
```
- The create_words_frequency function calculates the sum of occurrences of each word across all documents by summing the values along axis 0 in the feature matrix. This calculation provides an overall picture of how frequently each word appears in the entire collection of documents, which can help identify the most common and important words in the corpus.
- here feature_matrix can be tf-idf score or count-vectorizer matrix
- featuer name is the corresponding token  

```bash
def create_words_frequency(feature_matrix, features_name):
    features_df=pd.DataFrame(feature_matrix)
    features_df.columns=features_name
    sorted_features = features_df.sum(axis=0).sort_values(ascending=False) # sum along the rows (top to bottom for each column (token))
    sorted_features=pd.DataFrame(sorted_features)
    sorted_features=sorted_features.reset_index()
    sorted_features.columns=['Top Words','Counts']
    return sorted_features, features_df
```


  
### Compound Score 
In sentiment analysis, the compound score is a single value that represents the overall sentiment polarity and intensity of a piece of text. It's a normalized score that combines the positive, negative, and neutral sentiment scores to provide an overall sentiment evaluation. 


Values closer to -1 or 1 indicate stronger sentiment, while values closer to 0 indicate weaker sentiment or neutrality. For example:

- If a text has a compound score of 0.75, it suggests a strongly positive sentiment.
- If a text has a compound score of -0.5, it indicates a moderately negative sentiment.
- If a text has a compound score of 0.1, it suggests a slightly positive sentiment.


  
## 5.1 Vader by NLTK for Sentiment Analysis

<div align="center">
  <img src="https://nlp-sentiment.streamlit.app/~/+/media/9b739c83c3422163687542923a312b13f0045b196358fa192c712c56.png" alt="Image Alt Text" width="400">
</div>

- VADER (Valence Aware Dictionary and sEntiment Reasoner) is a sentiment analysis tool specifically designed for analyzing the sentiment of text data, especially social media text. It's a part of the Natural Language Toolkit (NLTK) library
- VADER is capable of understanding sentiment expressed in text and classifying it as positive, negative, or neutral, along with providing a sentiment intensity score (compound score) directly with its pipeline.
- Used its pipeline to compute compound score on "title" column of "US_Financial_News_Articles_2018.csv" dataframe.


```bash
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm
# Download the VADER lexicon (if not already downloaded)
nltk.download('vader_lexicon')
```

```bash
def compute_compound_score(text):
    
    '''Compute compound sentiment score using VADER SentimentIntensityAnalyzer.

    Parameters:
        text (str): The text for which you want to compute the compound score.

    Returns:
        float: The computed compound sentiment score.
    '''
    # Initialize the VADER SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    # Compute the compound sentiment score using VADER
    compound_score = sid.polarity_scores(text)['compound']

    return compound_score
```

## 5.2 Roberta Pre-Trained Model for Sentiment Analysis
<p align="left"> <a href="https://huggingface.co/roberta-base" target="_blank" rel="noreferrer">  <img src="https://huggingface.co/bertin-project/bertin-roberta-base-spanish/resolve/main/images/bertin.png" alt="Image Alt Text" width = "70"> </a>

- It is a module of the transformer library by Huggingface which enables reliable sentiment analysis for various types of English-language text. For each instance, it predicts either positive (2) or negative (0) sentiment or neutral (1) sentiment.


```bash
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL)
```
- softmax is used so that the sum of all 3 sentiments sum to 1. 

```bash
def polarity_score(example):
    #Set the return_tensors parameter to either pt for PyTorch, or tf for TensorFlow
    encoded_text = tokenizer(example,return_tensors='pt')
    output = model(**encoded_text)  # ** unpacks the dictionary
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    #Labels: 0 -> Negative; 1 -> Neutral; 2 -> Positive
    scores_dict = {'Negative': scores[0],
                   'Neutral' : scores[1],
                   'Positive' : scores[2]}
    return scores_dict
```

- this function is called while iterating through all 306240 (2 dropped) articles to compute sentiment analysis on 'title' column of the dataframe.
- after computing Positive, Negative and Neutral score for each artice it is stored in data frame with date as primary key.
- using the primary key an inner join is done on polarity__score dataframe and original dataframe of news articles.
- This dataframe is saved as US_financial_news_articles_2018_with_sentiment.csv
- Fixed the date formatting by indexing to just YYY-MM-DD and saved it to "my_sentiments.csv"

  
### Compound Score of Roberta and Vader modules

unlike Vader module, Roberta module does not have a pipeline to evaluate compound score. So, I used the following procedure to compute compound score from positive, negative and neutral sentiment scores of news articles.


```bash
import tqdm 
n = df.shape[0]
compound = {}
for i in tqdm.tqdm(range(0,n)):
    pos = df['Positive'][i]
    neg = df['Negative'][i]
    neu = df['Neutral'][i]
    id = df['index'][i]
    compound[id] = float(pos - neg / 1 - neu )

print(compound)
```
- Stored compund score in a dict with index as key and compound score as value.
- Thereafter, Normalized scores in the range -1 to 1.
- Converted dict to datarame
- Merged US_financial_news_articled _with _sentiments.csv with compound score(Roberta) dataframe with index as primary key.
- Added compound score(Vader) to the resultant dataframe and saved it as "Vader_Roberta_sentiment.csv".

### Splitting 

- Grouped the resultant dataframe with respect to "source_name"

```bash
df_3 = pd.read_csv('/Users/khushal/Desktop/Projects/Vader_Roberta_sentiment.csv')
df_3['source_name'].unique()
groups = df_3.groupby(['source_name']).size()
print(groups)

#source_name
#cnbc.com        85196
#fortune.com      5737
#reuters.com    197512
#wsj.com         17793
#dtype: int64
```
- Splitted dataframe using these unique source names to form 4 dataframes, namely, 'cnbc_sentiment.csv','fortune_sentiment.csv','reuters_sentiment.csv','wsj_sentiments.csv'


```bash
#df_3_wsj shape: (17793, 12)
#df_3_cnbc shape: (85196, 12)
#df_3_fortune shape: (5737, 12)
#df_3_reuters shape: (197512, 12)
```

- In these datarames there are multiple articles published on the same date.
- So, computed aggregate mean with respect to unique publishing dates creatig the following dataframes.


```bash
df_4_wsj = df_3_wsj.groupby(['published_date']).agg(['mean'])
df_4_wsj = df_4_wsj.reset_index()  # to bring 'published_date' back to main columns
print(df_4_wsj.shape)
df_4_cnbc = df_3_cnbc.groupby(['published_date']).agg(['mean'])
df_4_cnbc = df_4_cnbc.reset_index()
print(df_4_cnbc.shape)
df_4_fortune =df_3_fortune.groupby(['published_date']).agg(['mean'])
df_4_fortune = df_4_fortune.reset_index()
print(df_4_fortune.shape)
df_4_reuters = df_3_reuters.groupby(['published_date']).agg(['mean'])
df_4_reuters = df_4_reuters.reset_index()
print(df_4_reuters.shape)

#(165, 8)
#(159, 8)
#(155, 8)
#(159, 8)
```
### Source_Price

- snp.csv is a dataframe which has 121 rows and 7 columns.

```bash
RangeIndex: 121 entries, 0 to 120
Data columns (total 7 columns):
 #   Column     Non-Null Count  Dtype         
---  ------     --------------  -----         
 0   Date       121 non-null    datetime64[ns]
 1   Open       121 non-null    float64       
 2   High       121 non-null    float64       
 3   Low        121 non-null    float64       
 4   Close      121 non-null    float64       
 5   Adj Close  121 non-null    float64       
 6   Volume     121 non-null    int64         
dtypes: datetime64[ns](1), float64(5), int64(1)
```

- Which means there should be 121 unique publishing dates for each dataframe unique to source_name.
- Therefore, we use the following function to create new table which combines the compound scores with publication dates in 'snp.csv'.

```bash

def new_table(df_new, snp, df):
    snp_length = snp.shape[0]
    df_len = df.shape[0]
    df = df.copy()
    
    # Convert the 'Date' columns to Timestamp format
    snp['Date'] = pd.to_datetime(snp['Date'])
    df[('published_date', )] = pd.to_datetime(df[('published_date', )])
    
    for i in range(snp_length):
        idx = i
        date = snp.at[i, 'Date']  # Access the date at index i
        j = 0
        t = 0
        
        while j < df_len:
            if date == df.loc[j, ('published_date', )]:  # Use .loc with tuple to access multi-level column
                mean_compound = df.loc[j, ('compound_score_Roberta', 'mean')]  # Access mean_compound column value
                comp_flag = 1
                t = 1
                break
            j += 1
        
        if t == 0:
            mean_compound = 0
            comp_flag = 0

        df_new = df_new.append({'idx': idx,
                                'date': date,
                                'mean_compound': mean_compound,
                                'comp_flag': comp_flag},
                               ignore_index=True)
        
    return df_new



data_wsj = new_table(df_new_wsj, snp, df_4_wsj)
data_fortune = new_table(df_new_fortune, snp, df_4_fortune)
data_cnbc = new_table(df_new_cnbc, snp, df_4_cnbc)
data_reuters = new_table(df_new_reuters, snp, df_4_reuters)
```

- The code iterates through rows in the "snp" DataFrame, retrieving a date and initializing a search index "j" for the "df" DataFrame. It then compares the date from "snp" with dates in "df" using a nested loop, updating a "mean_compound" value and a flag if a match is found.
-  Finally, the code constructs a new DataFrame "df_new" by appending rows containing the date, calculated sentiment information, and a flag indicating match status for each date in "snp".
- A flag t is set to 0 to track whether a match was found.
- comp_flag = 1 means publication date is the same in snp.csv and dataframe safter splitting based on source_name.
- comp_flag = 0 means publication date is not same in snp.csv and dataframe safter splitting based on source_name.
- Finally we concatenate new dataframes with Adj_Close from snp

```bash
  source_price = pd.concat([data_wsj['date'],data_wsj['wsj_mean_compound'],data_cnbc['cnbc_mean_compound'],data_fortune['fortune_mean_compound'],data_reuters['reuters_mean_compound'], snp['Adj Close']], axis=1)

```

- It is saved as source_price.csv --> with shape (121, 6)

### Stock Data from 2017-12-07 to 2018-06-01
- This range of dates corresponds to source_price.csv and 455 other dataframes from S&P 500 list after cutting.
- As we know original datset for 455 companies enlisted in S&P 500 list range from 2007 to 2018.
- Therefore, we cut the data and transform it such a way that its attributes(dates) match the source_price.csv so thta we will have a final dataset for running our LSTM model.

```bash
 # row: 2537 corresponds to date -> 2017-12-07, and 2657 -> 2018-06-01
    df_cut = pd.DataFrame(df_new.iloc[2537:2658, :])
```
  
This final dataset will have the following columns:

- Date
- Price
- wsj_mean_compound
- fortune_mean_compound
- reuters_mean_compound
- cnbc_mean_compound
Here stock price will be diffrent for each company enlisted i S&P 500 list. Thats why we will be having 455 such csv files as our final dataset.

## LSTM Model

- Before feeding the data into LSTM model we first split data into test and train.
- We begin by extracting paths of each csv file ( all 455 files ) and iterate over them.
- compute variance. The higher the variance a number has, the more separated it is from the mean.
- Then we add noise to the data in a fashion such that 1 out of 4 attributes have noise. Therefore, there will be 4 such combinations ( wsj , fortune, cnbc, reuters ) created using create_table function. ( mu is 0 for now )
- Pick random samples from Gaussian distribution.
- noise = 0.1


<div align="center">
  <img src="https://d4y70tum9c2ak.cloudfront.net/contentImage/5TZF1HawG508EFGaoHu98UndSkfOQJLQ8hnAXsAdZTI/resized.png" alt="Image Alt Text" width = "500">
</div>
 
```bash
    # Variance = Σ ((xi - mean) ** 2) / N
    # Variance measures how much the values in a dataset vary from the mean. It is a measure of the spread or dispersion of the data points.
    wsj_var = np.var(df['wsj_mean_compound'])
    cnbc_var = np.var(df['cnbc_mean_compound'])
    fortune_var = np.var(df['fortune_mean_compound'])
    reuters_var = np.var(df['reuters_mean_compound'])

    # Adding noise
    sigma_wsj = noise * wsj_var
    sigma_cnbc = noise * cnbc_var
    sigma_fortune = noise * fortune_var
    sigma_reuters = noise * reuters_var

    df_noise = pd.DataFrame()
    df_noise['wsj_noise'] = df['wsj_mean_compound']
    df_noise['cnbc_noise'] = df['cnbc_mean_compound']
    df_noise['fortune_noise'] = df['fortune_mean_compound']
    df_noise['reuters_noise'] = df['reuters_mean_compound']
    #print('Here', df)

    # np.random.normal(): used to generate random samples from a normal (Gaussian) distribution.
    for i in range(0,df.shape[0]):
        df_noise['wsj_noise'] += np.random.normal(mu, sigma_wsj)
        df_noise['cnbc_noise'] += np.random.normal(mu, sigma_cnbc)
        df_noise['fortune_noise'] += np.random.normal(mu, sigma_fortune)
        df_noise['reuters_noise'] += np.random.normal(mu, sigma_reuters)
    df_n = df_noise

    # create_table(df_1, df_2, df_3, df_4, column_1, column_2, column_3, column_4):
    df_1n = create_table(df_n, df, df, df,df, 'wsj_noise', 'cnbc_mean_compound', 'fortune_mean_compound', 'reuters_mean_compound', 'price') # (121,5)
    df_2n = create_table(df, df_n, df, df,df, 'wsj_mean_compound', 'cnbc_noise', 'fortune_mean_compound', 'reuters_mean_compound', 'price') # (121,5)
    df_3n = create_table(df, df, df_n, df,df, 'wsj_mean_compound', 'cnbc_mean_compound', 'fortune_noise', 'reuters_mean_compound', 'price') # (121,5)
    df_4n = create_table(df, df, df, df_n,df, 'wsj_mean_compound', 'cnbc_mean_compound', 'fortune_mean_compound', 'reuters_noise', 'price') # (121,5)
```

- Then we split the data into 85% train and 15% test set.
- Followed by Normalizing the stock price of each dataset combination having noise. This is done essentially to reduce the computation.
- In the function normalize_data, we do a simple sliding window with 10 time stamps in a single window.
- In this window after normalizing we trasnpose it and further split into x_train, y_train and x_test, y_test. Such that the shape is as follows:-


```bash
def normalise_data(data_train, sequence_length):  # data_train is numpy array of shape(102,5)
    record_min = []
    record_max = []
    data_windows = []
    for i in range(len(data_train) - sequence_length):
        data_windows.append(data_train[i:i + sequence_length]) # sequence data at every 10 day timestamp
    data_windows = np.array(data_windows).astype(float)
```

- further splitting time stamps

```bash
x_train = normalised_data[:, :-1] #---> prices till 9 time stamps
y_train = normalised_data[:, -1, [0]] #---> price to predict after 9th time stamp
```


```bash
# x_train_1.shape --> (92,9,5)--|
# x_train_2.shape --> (92,9,5)--|
# x_train_3.shape --> (92,9,5)--|---> concatenate----> x_train.shape = (368,9,5)
# x_train_4.shape --> (92,9,5)--|


# y_train_1.shape --> (92,1)--|
# y_train_2.shape --> (92,1)--|
# y_train_3.shape --> (92,1)--|---> concatenate----> y_train.shape = (368,1)
# y_train_4.shape --> (92,1)--|
```
- LSTM model

```bash
# parameters-------------
split = 0.85
sequence_length = 10
batch_size = 100
input_dim = 5
input_timesteps = 9
neurons = 50
epochs = 5
prediction_len = 1
dense_output = 1
drop_out = 0.2
# LSTM Model--------------
model = Sequential()
model.add(LSTM(neurons, input_shape = (input_timesteps, input_dim), return_sequences = True))
model.add(Dropout(drop_out)) # --> to prevent vanishing or exploding gradient problem
model.add(LSTM(neurons, return_sequences = True))
model.add(LSTM(neurons, return_sequences = False))
model.add(Dropout(drop_out)) # --> to prevent vanishing or exploding gradient problem
model.add(Dense(dense_output, activation='linear'))
```
- Used Adam optimizer and mean_squared_error as loss function 
- Adam optimizer adapts the learning rate for each parameter, which helps accelerate convergence
  
```bash
# Compile Model
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
```
- Followed by fit

```bash
model.fit(x_train, y_train, batch_size, epochs )
```

### Predictions 
- model.predict is used inside a function to loop over all the windows of x_test --> (9,9,5)
- current_frame is reshaped to (1, input_timesteps, input_dim) using newaxis, which essentially means that you are providing a single input example with input_timesteps time steps and input_dim features. The model then predicts the output for this single example.
- these predictions are stored in a list.
- Then we slide the left pointer of the sliding wondow (current frame) to predict value of next sequence. 
- This predicted value is stored at 9th index of the sequence in x_test. and window is slided simultaneous and the loop goes on untill 9 predictions are evaluated and inserted for each stock company. 

```bash
  # Prediction
    data = x_test
    prediction_sequence = []
    pre_win_num = int(len(data)/prediction_len)
    window_size = sequence_length

    for i in range(0, pre_win_num):
        current_frame = data[i * prediction_len]
        predicted = []
        for j in range(0, prediction_len):
            # By using np.newaxis, we effectively increase the dimensionality of curr_frame from a 2D array
            # (shape: (input_timesteps, input_dim)) to a 3D array (shape: (1, input_timesteps, input_dim)).
            # This is done to match the expected shape for prediction with the LSTM model.
            temp = model.predict(current_frame[newaxis, :, :])[0]
            # the output of model.predict() is a 2D array with shape (1, dense_output)
            #  The [0] indexing is used to extract the predicted value from the 2D array and convert it to a scalar.
            predicted.append(temp)
            # after predicting the value at the current time step,
            # we need to shift the input sequence by one step to include the newly predicted value and exclude the oldest value from the sequence.
            current_frame = current_frame[1:]
            # Sliding window --> [ window_size - 1 ]
            # new_arr = np.insert(arr, index, row_to_insert, axis=0)
            # new_arr = np.insert(arr, index, column_to_insert, axis=1)
            current_frame = np.insert(current_frame, [window_size - 1], predicted[-1], axis = 0 )
        prediction_sequence.append(predicted)
```

### Denormalization 

- Then we denormalize the predicted prices as we normalized them earlier to make computation easy.

```bash
de_predict.append(prediction_sequence[i][j][0] * record_max[m] + record_min[m])
```

- Then we evaluate error and find accuracy using y_test_original.

<div align="center">
  <img src = "https://suboptimal.wiki/images/mse_5.jpg" alt= "Image Alt Text" width = "380"> 
</div>

 
- Here there will be 9 predictions for each stock company.
- So, there will be 1 accuracy for each stock company.

```bash
 # Error
    error = []
    squared_error = []
    absolute_error = []
    error_percent = []
    daily_accuracy = []
    diff = y_test.shape[0] - prediction_len * pre_win_num
    for i in range(y_test_ori.shape[0] - diff):
        error.append(y_test_ori[i,] - de_predict[i])

        # Calculate squared and absolute error for each data point
        squared_error.append(error[-1] * error[-1])
        absolute_error.append(abs(error[-1]))

        # Calculate and append accuracy for each day
        val = absolute_error[-1] / y_test_ori[i,]
        val = abs(val)
        error_percent.append(val)
        daily_mean_error_percent = sum(error_percent) / len(error_percent)
        daily_accuracy.append(1 - daily_mean_error_percent)

    # Calculate overall accuracy and MSE
    overall_accuracy = sum(daily_accuracy) / len(daily_accuracy)
    overall_MSE = sum(squared_error) / len(squared_error)
```

## Stock Predictions

- Here is the prediction of last 9 days with (using compoud score estimated from sentiment analysis of news articles) and without compoud score.

<div align="center">
  <img src="https://github.com/khushals025/Stock_Prediction_Using_Financial_News/blob/main/snp500.jpg?raw=true" alt="Image Alt" width ="2000">
</div>
    
## Accuracy

- Accuracy Comparison
<div align="center">
  <img src="" alt="Image Alt" width ="2000">
</div>
  
## Results

## Future Scope


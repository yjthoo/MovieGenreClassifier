import numpy as np
import pandas as pd
from ast import literal_eval

import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def preprocessOverview(text):
    
    # tokenize text and set to lower case
    tokens = [x.strip().lower() for x in nltk.word_tokenize(text)]
    
     # get stopwords from nltk, remove them from our list as well as all punctuations and numbers
    stop_words = stopwords.words('english')
    output = [word for word in tokens if (word not in stop_words and word.isalpha())]
    
    return " ".join(output)


def preprocessGenres(df, relevant_treshold):
    # extract the names of the genres
    df["genres"] = df["genres"].fillna('[]').apply(literal_eval).apply(lambda x: [i["name"] for i in x] if isinstance(x, list) else [])

    # drop rows where no genre is specified # (45'466 --> 43'024)
    df.drop(df.loc[(df['genres'].str.len() == 0),:].copy().index, inplace = True)
    
    # keep relevant genres (more than 1000 appearances) (32 genres to 19)
    all_genres = df["genres"].tolist()
    full_list_genres = [genre for movie_genres in all_genres for genre in movie_genres] 
    freqdist = FreqDist(full_list_genres)
    relevant = sorted(w for w in set(full_list_genres) if freqdist[w] > relevant_treshold) 
    
    # keep only one genre per movie
    def set_genre(serie, relevant):
        for genre in serie:
            if genre in relevant:
                return genre
        return np.nan
    df["genres"] = df["genres"].apply(lambda x: set_genre(x, relevant))
    
    # remove movies without relevant genre (43024, 5) --> (43014, 5)
    df = df.dropna()
    
    return df


def genresLabel(df):
    
    # determine the unique genres contained within our dataset
    genres = df["genres"]
    genres = sorted(set(genres))
    
    genresDic = {}
    labelNb = 0
    
    # assign a label to each genre
    for genre in genres:
        genresDic[genre] = labelNb
        labelNb += 1
        
    # create a column and insert the value corresponding to the genre
    df["genre label"] = df["genres"].apply(lambda x: genresDic[x])
    
    return df, genresDic


def preprocessDataset(df, relevant_treshold=1000):
    
    # keep only the columns that are relevant to this application
    relCols = ['original_title', 'overview', 'genres']
    df = df[relCols].copy()      #(45466, 24) --> (45466, 3)
    
    # preprocess overview
    df["overview"] = df["overview"].astype(str).apply(lambda x: preprocessOverview(x))
    
    # preprocess genres label
    df = preprocessGenres(df, relevant_treshold)
    
    # give genres a label
    df, genresDict = genresLabel(df)
    
    return df, genresDict


def labelFile(genresDict):
    genreLabel = pd.DataFrame.from_dict(genresDict, orient='index')
    genreLabel.reset_index(level=0, inplace=True)
    genreLabel.columns = ['genre', 'label']
    return genreLabel



if __name__ == '__main__':

    input_file = 'datasets/movies_metadata.csv'
    print("Pre-processing dataset from " + input_file)
    df = pd.read_csv(input_file)
    # Pre-processing dataframe
    df, genresDict = preprocessDataset(df, relevant_treshold=1000)
    save_df_path = 'datasets/preprocessed.csv'
    print("Saving pre-processed data at " + save_df_path)
    df.to_csv(save_df_path, index=False)

    # Create file for user label number to associated movie
    genreLabel = labelFile(genresDict)
    save_label_path = 'datasets/genreLabels.csv'
    print("Saving labels at " + save_label_path)
    genreLabel.to_csv(save_label_path, index = False)
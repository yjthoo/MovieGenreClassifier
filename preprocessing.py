import numpy as np
import pandas as pd
from ast import literal_eval

import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def preprocessOverview(text):

    """
    Function that preprocesses sentence by setting to lower case, removing stop words and removing 
    words that contain characters that are not letters
    
    Arguments:
    text -- overview of the movie 

    Returns:
    output -- preprocessed overview
    """
    
    # tokenize text and set to lower case
    tokens = [x.strip().lower() for x in nltk.word_tokenize(text)]
    
     # get stopwords from nltk, remove them from our list as well as all punctuations and numbers
    stop_words = stopwords.words('english')
    output = [word for word in tokens if (word not in stop_words and word.isalpha())]
    
    return " ".join(output)


def preprocessGenres(df, relevant_treshold):

    """
    Function that preprocesses genres of the movie by removing the movies that do not contain a genre
    and setting only one genre for each movie (usually the first).

    In the case where the first genre does not have many appearances, it finds the next most relevant genre

    Arguments:
    df -- movies dataframe
    relevant_treshold -- threshold for the minimum number of appearances a genre must have 

    Returns:
    df -- movies dataframe to which the preprocessing has been applied
    """

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

    """
    Function that sets a label to each genre

    Arguments:
    df -- movies dataframe

    Returns:
    df -- movies dataframe which a column with the label corresponding to the genre
    genresDic -- dictionnary mapping each genre to its corresponding label
    """
    
    # determine the unique genres contained within our dataset and sort them alphabetically
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

    """
    Function that cleans the initial dataset by removing unncessary columns, calling the functions presented above
    and removing non valid datapoints

    Arguments:
    df -- movies dataframe

    Returns:
    df -- preprocessed dataframe
    genresDic -- dictionnary mapping each genre to its corresponding label
    """
    
    # keep only the columns that are relevant to this application
    relCols = ['original_title', 'overview', 'genres']
    df = df[relCols].copy()      #(45466, 24) --> (45466, 3)
    
    # preprocess overview
    df["overview"] = df["overview"].astype(str).apply(lambda x: preprocessOverview(x))
    
    # determine the length of the movie overview
    df['overview length'] = df['overview'].apply(lambda x: len(str(x).split(' ')))
        
    # preprocess genres label and keep only the genres that have over 1000 appearances
    df = preprocessGenres(df, relevant_treshold)
    
    # give genres a label
    df, genresDict = genresLabel(df)
    
    df.dropna(inplace = True)
    
    return df, genresDict


def labelFile(df, genresDict):

    """
    Function that creates a file to map the genre to its corresponding label and determine the class weights

    Arguments:
    df -- preprocessed movies dataframe
    genresDict -- dictionnary mapping each genre to its corresponding label

    Returns:
    genreLabel -- dataframe mapping each genre to its corresponding label and displaying the weight of each class
    """

    # create a dataframe with the mapping from genre to corresponding label
    genreLabel = pd.DataFrame.from_dict(genresDict, orient='index')
    genreLabel.reset_index(level=0, inplace=True)
    genreLabel.columns = ['genre', 'label']
    
    # determine the number of times each genre is contained within the dataset
    genreCol = df['genres'].tolist()
    fdist1 = FreqDist(genreCol)
    
    class_weights = []

    # determine the weight of each genre class
    for key, value in sorted(fdist1.most_common(50)):
        class_weights.append(value/len(df))
        
    genreLabel['weight'] = class_weights
    
    return genreLabel



if __name__ == '__main__':

    # read data from the input file
    input_file = 'datasets/movies_metadata.csv'
    print("Pre-processing dataset from " + input_file)
    df = pd.read_csv(input_file, low_memory=False)

    # Pre-process dataframe and save file 
    df, genresDict = preprocessDataset(df, relevant_treshold=1000)
    save_df_path = 'datasets/preprocessed.csv'
    print("Saving pre-processed data at " + save_df_path)
    df.to_csv(save_df_path, index=False)

    # Create a file to map the genre to its corresponding label and determine the class weights
    genreLabel = labelFile(df, genresDict)
    save_label_path = 'datasets/genreLabels.csv'
    print("Saving labels at " + save_label_path)
    genreLabel.to_csv(save_label_path, index = False)
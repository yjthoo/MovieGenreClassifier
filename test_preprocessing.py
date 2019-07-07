import sys
import pandas as pd
from pandas.util.testing import assert_frame_equal

from preprocessing import preprocessOverview, preprocessDataset, genresLabel, labelFile

def test_preprocessOverview():
    testSentence = " I am doing a Project for MoVie's Genre . . Classification , :D   and it is sunny 3 4 93 here; "
    expectedResult = "project movie genre classification sunny"
    result = preprocessOverview(testSentence)
    assert expectedResult == result, "preprocessOverview() outputs unexpected string"


def test_preprocessDataset():
    testDf = pd.read_csv("test_data/test_input_head10.csv")
    expectedResultDf = pd.read_csv("test_data/result_test_thresh3_head10.csv")
    
    resultDf, resultGenresDic = preprocessDataset(testDf, relevant_treshold=3)
    assert_frame_equal(expectedResultDf, resultDf)


def test_genresLabel():
    df = pd.read_csv("test_data/result_test_thresh3_head10.csv")
    resultDf, resultGenresDict = genresLabel(df)
    expectedResultGenresDict = {'Action': 0, 'Adventure': 1, 'Comedy': 2}
    
    assert_frame_equal(df, resultDf)
    assert resultGenresDict == expectedResultGenresDict, "test_genresLabel() outputs unexpected dictionnary"


def test_labelFile():
    test_df = pd.read_csv("test_data/test_input_head10.csv")
    expected = {'genre': ['Action', 'Adventure', 'Comedy'], 'label': [0, 1, 2], 'weight': [0.3, 0.2, 0.5]}
    expectedResultGenreLabel = pd.DataFrame(data=expected)
    
    df, genresDict = preprocessDataset(test_df, relevant_treshold=3)
    genreLabel = labelFile(df, genresDict)
    assert_frame_equal(expectedResultGenreLabel, genreLabel)

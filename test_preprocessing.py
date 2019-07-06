import sys
import pandas as pd
from pandas.util.testing import assert_frame_equal

from preprocessing import preprocessOverview, preprocessDataset, genresLabel

def test_preprocessOverview():
    test_sentence = " I am doing a Project for MoVie's Genre . . Classification , :D   and it is sunny 3 4 93 here; "
    expected_result = "project movie genre classification sunny"
    result = preprocessOverview(test_sentence)
    assert expected_result == result, "preprocessOverview() outputs unexpected string"


def test_preprocessDataset():
    test_df = pd.read_csv("test_data/test_input_head10.csv")
    expected_result_df = pd.read_csv("test_data/result_test_thresh3_head10.csv")
    
    result_df, result_genresDic = preprocessDataset(test_df, relevant_treshold=3)
    assert_frame_equal(expected_result_df, result_df)


def test_genresLabel():
    df = pd.read_csv("test_data/result_test_thresh3_head10.csv")
    result_df, result_genresDict = genresLabel(df)
    expected_result_genresDict = {'Action': 0, 'Adventure': 1, 'Comedy': 2}
    
    assert_frame_equal(df, result_df)
    assert result_genresDict == expected_result_genresDict, "test_genresLabel() outputs unexpected dictionnary"
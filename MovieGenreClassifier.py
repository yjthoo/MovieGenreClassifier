#!/usr/bin/python

import numpy as np 
import sys, getopt
import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model
from trainModel import read_glove_vecs_only_alpha, sentences_to_indices
from preprocessing import preprocessOverview


# import only system from os 
from os import system, name 

# define our clear function, taken from: https://www.geeksforgeeks.org/clear-screen-python/
def clear(): 
  
    # for windows 
    if name == 'nt': 
        _ = system('cls') 
  
    # for mac and linux(here, os.name is 'posix') 
    else: 
        _ = system('clear') 

def predictGenre(overview):

	# load the model
	model = load_model('models/train-0.56_validation-0.40.h5')

	df = pd.read_csv('datasets/preprocessed.csv')
	df.dropna(inplace = True)

	genreLabels = pd.read_csv('datasets/genreLabels.csv')

	# obtain the GloVe dataset of dimensionality 100
	word_to_index, index_to_word, word_to_vec_map = read_glove_vecs_only_alpha('datasets/glove.6B/glove.6B.100d.txt')

	# determine the maximum length of a movie overview
	max_sequence_length = df["overview length"].max()

	# preprocess the input and convert to indices
	testOverview = []
	testOverview.append(preprocessOverview(overview))
	overview_indices = sentences_to_indices(np.asarray(testOverview), word_to_index, max_sequence_length)

	# predict genre label and map back to genre string
	prediction = model.predict(overview_indices)

	confidence = prediction.max()
	predictedGenre = genreLabels[genreLabels["label"] == np.argmax(prediction)]["genre"].values[0]

	return predictedGenre, confidence


def main(argv, clearConsole = True, outputConfidence = False):

	inputTitle = ''
	inputDescription = ''
	validArguments = ['title=', 'description=']

	try:
		# parse command line options
		opts, args = getopt.getopt(argv, '', validArguments)
	except getopt.GetoptError:
		print('The input should have the following structure: MovieGenreClassifier.py --title <title> --description <description>')
		sys.exit(2)

	try:
		# ensure that there is the correct amount of input arguments
		assert len(opts) == len(validArguments) and len([option[0] for option in opts if option[0] in ["--title", "--description"]]) == len(validArguments)
	except AssertionError:
		raise AssertionError("Invalid argument or arguments! The input should have the following structure: MovieGenreClassifier.py --title <title> --description <description>")
		sys.exit()

	# iterate through the input arguments
	for opt, arg in opts:

		if opt == "--title":
			inputTitle = arg
		else:

			# ensure that the description argument is comprised of strings
			try:
				assert not arg.isnumeric()
			except AssertionError:
				raise AssertionError(str(opt) + " argument needs to be a string")
				sys.exit()

			inputDescription = arg

	movieDic = {}
	movieDic["title"] = inputTitle
	movieDic["description"] = inputDescription
	movieDic["genre"] = "unkown"
	movieDic["genre"], confidence = predictGenre(inputDescription)

	# clear console before printing
	if clearConsole:
		clear()

	print(movieDic)

	if outputConfidence:
		print("\n The model had a confidence of " + str(confidence) + " on this prediciton.")




if __name__ == "__main__":
   main(sys.argv[1:], clearConsole = True, outputConfidence = True)
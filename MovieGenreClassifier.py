#!/usr/bin/python

import numpy as np 
import sys, getopt
import tensorflow as tf
import keras
from trainModel import read_glove_vecs_only_alpha, sentences_to_indices

def predictGenre(overview):

	model = load_model('.h5')

	df = pd.read_csv('datasets/preprocessed.csv')
	df.dropna(inplace = True)

	genreLabels = pd.read_csv('datasets/genreLabels.csv')

	# obtain the GloVe dataset of dimensionality 100
	word_to_index, index_to_word, word_to_vec_map = read_glove_vecs_only_alpha('datasets/glove.6B/glove.6B.100d.txt')

	# determine the maximum length of a movie overview
	max_sequence_length = df["overview length"].max()

	X_test = sentences_to_indices([overview], word_to_index, max_sequence_length)

	# predict genre label and map back to genre string
	prediction = model.predict(X_test)
	prediction = prediciton.index(1.0)
	predictedGenre = genreLabels[genreLabels["label"] == prediction]["genre"]

	return predictedGenre


def main(argv):

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
	#movieDic["genre"] = predictGenre(inputDescription)
	print(movieDic)

if __name__ == "__main__":
   main(sys.argv[1:])
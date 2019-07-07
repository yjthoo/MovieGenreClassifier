#!/usr/bin/python

import numpy as np 
import sys, getopt

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
	print(movieDic)

if __name__ == "__main__":
   main(sys.argv[1:])
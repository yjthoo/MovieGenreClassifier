#!/usr/bin/python

import numpy as np 
import sys, getopt

def main(argv):
	inputTitle = ''
	inputDescription = ''

	try:
		opts, args = getopt.getopt(argv, '', ['title=', 'description='])
	except getopt.GetoptError:
		print('MovieGenreClassifier.py --title <title> --description <description>')
		sys.exit(2)

	#iterate through the input arguments
	for opt, arg in opts:

		if opt == '-h':
			print('MovieGenreClassifier.py --title <title> --description <description>')
			sys.exit()
		elif opt in ["--title", "--description"]:

			try:
				assert not arg.isnumeric()
			except AssertionError:
				raise AssertionError(str(opt) + " argument needs to be a string")
				sys.exit()

			if opt == "--title":
				inputTitle = arg
			else:
				inputDescription = arg

	movieDic = {}
	movieDic["title"] = inputTitle
	movieDic["description"] = inputDescription
	movieDic["genre"] = "unkown"
	print(movieDic)

if __name__ == "__main__":
   main(sys.argv[1:])
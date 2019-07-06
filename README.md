# MovieGenreClassifier

The goal of this project is to determine the genre of a movie based on its title as well as its description/overview (generally up to 140 words). The program is written in Python and the model currently uses the following architecture: 


The program then enables the classification between the following 23 types of genres:



## Installation

You will first need to clone this repository using the following command in a terminal:


```
git clone https://github.com/yjthoo/MovieGenreClassifier.git
```

The following lines will show you how to install python as well as the following required libraries:

* [TensorFlow](https://www.tensorflow.org/):
* [Matplotlib](https://matplotlib.org/): This library is used to generate plots to analyse the performance of the model used to predict the genre of the movie based on its overview. 
* [pandas](https://pandas.pydata.org/):
* [Keras](https://keras.io/):
* [scikit-learn](https://scikit-learn.org/stable/):
* [NLTK](https://www.nltk.org/):

In this section, we will cover how to install [anaconda](https://www.anaconda.com/) as this will provide you with the [Jupyter Notebook](https://jupyter.org/) IDE (Integrated Development Environment) that was used to analyse/preprocess the data for this project.

1. In the following [link](https://www.anaconda.com/distribution/), download the open-source `Python 3.7 version` Anaconda Distribution (available for Linux, Windows, and Mac OS X). 

2. Follow the installation guide corresponding to your OS in the following [link](https://docs.anaconda.com/anaconda/install/) (**Note that the first step of the installation (downloading the installer) has already been done in the previous step**).

3. Next, we will install TensorFlow by creating a virtual envinronment as shown in the following [link](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html) which illustrates how to install either a CPU or GPU version of TensorFlow based on your computer. 

4. Once this is done, please make sure that your environment is still active. You can simply check this by verrifying the name displayed within brackets in your *Anaconda Prompt* window.
	For instance:
	```
	(environment_name) C:\Users\me>
	```

	where `environment_name` is the name of the environment that you created indicates that your environment is active. If this is not the case, simply type:
	```
	activate environment_name
	```

	This will ensure that all following libraries will be installed in this environment. 

5. The next step consists of installing Matplotlib, pandas, Keras and scikit-learn by typing the following lines in the command prompt:
	```
	conda install matplotlib
	conda install pandas
	conda install keras
	conda install scikit-learn
	```

	or in case of any issues, please follow the installation guides of the respective libraries:
	* [Matplotlib](https://matplotlib.org/users/installing.html)
	* [pandas](https://pandas.pydata.org/pandas-docs/stable/install.html)
	* [Keras](https://keras.io/#installation)
	* [scikit-learn](https://scikit-learn.org/stable/install.html)

6. The final stage of the installation guide consists of the installation of NLTK with the following command:
	```
	conda install nltk
	```

	In addition to this, you will also need to do the following as this project makes use of NLTK's stopwords:

	1. In the command prompt, type:
		```
		python
		```

	This will allow you to directly run Python commands in the terminal.

	2. Type the following commands:
		```
		import nltk
		nltk.download()
		```

	This will open a new window and enable you to download the collections by clicking on download. You should obtain something similar to the following image:

	![alt text](images/NLTK_installer.png "NLTK installer")

	3. Once this is done, you can simple type:
		```
		quit()
		```

	which will stop the Python interpreter. 


## Datasets

In addition to the libraries listed above, you will need to download the following datasets:


## Running the program

1. Open an *Anaconda Prompt* window and activate your virtual environment. 

## Modifying the program

To open the notebooks used to preprocess the data and test each stage of the program, you will need to ensure that your virtual environment is active when opening Jupyter Notebook
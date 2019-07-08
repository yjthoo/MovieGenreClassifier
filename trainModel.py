import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
np.random.seed(1)


def read_glove_vecs_only_alpha(glove_file):

	"""
	Adapted from Deep Learning Specialization by deeplearning.ai: https://www.coursera.org/specializations/deep-learning?

    Obtains the GloVe vectors of the words that only contain alphabetical letters
    
    Arguments:
    glove_file -- path to the GloVe dataset

    Returns:
    word_to_index -- dictionary mapping from words to their indices in the vocabulary
    index_to_words -- dictionary mapping indices to their corresponding words
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation
    """

    with open(glove_file, 'r',encoding='utf8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            
            # only consider words containing alphabetical letters
            if curr_word.isalpha():
                words.add(curr_word)
                word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
            
    return words_to_index, index_to_words, word_to_vec_map


def sentences_to_indices(X, word_to_index, max_len):

    """
	Adapted from Deep Learning Specialization by deeplearning.ai: https://www.coursera.org/specializations/deep-learning?  

    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can betest = pd.read_csv('datasets/genreLabels.csv') given to `Embedding()` 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    
    # number of training examples
    m = X.shape[0]                                   
    
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))
    
    # loop over training examples
    for i in range(m):                               
        
        # Convert the ith training sentence in lower case and split is into words -> get a list of words.
        sentence_words = [x.lower() for x in X[i].split()]
        
        # Initialize j to 0
        j = 0
        
        # Loop over the words in sentence_words
        for w in sentence_words:
            
            # check that the word is within our GloVe dataset, otherwise pass
            if w in word_to_index.keys():
                # Set the (i,j)th entry of X_indices to the index of the correct word.
                X_indices[i, j] = word_to_index[w]
                
                # Increment j to j + 1
                j = j+1
            else:
                pass
                
    return X_indices


def pretrained_embedding_layer(word_to_vec_map, word_to_index):

    """
	Adapted from Deep Learning Specialization by deeplearning.ai: https://www.coursera.org/specializations/deep-learning?

    Creates a Keras Embedding() layer and loads in pre-trained GloVe 100-dimensional vectors.
    
    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """
    
    # adding 1 to fit Keras embedding (requirement)
    vocab_len = len(word_to_index) + 1    
    
    # define dimensionality of your GloVe word vectors (in our case 100)
    emb_dim = word_to_vec_map["cucumber"].shape[0]      
    
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes
    embedding_layer = Embedding(vocab_len, emb_dim, trainable = False) 

    # Build the embedding layer, required before setting the weights of the embedding layer
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


def GenreClassifierV2(input_shape, word_to_vec_map, word_to_index, nbClasses, dropout_rate = 0.2):

    """
    Function creating the graph of the model
    
    This model was inspired by the model presented in: https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras 
    """
    
    # Define input of the graph of dtype 'int32' as it contains indices
    sentence_indices = Input(shape = input_shape, dtype = 'int32')
    
    # Create the embedding layer pretrained with GloVe Vectors 
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)
    
    # Add 1-dimensional convolutionnal layer with same padding
    X = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(embeddings)
    
    # Max pooling
    X = MaxPooling1D(pool_size=2)(X)
    
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # output is a batch of sequences
    X = LSTM(100, return_sequences = False)(X)

    # Add dropout with a probability of 0.5
    X = Dropout(dropout_rate)(X)
    
    # Propagate X through a Dense layer with softmax activation to get back a batch of 23-dimensional vectors.
    X = Dense(nbClasses)(X)

    logits = X
    
    # Add a softmax activation
    X = Activation('softmax')(X)

    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)
        
    return model, logits


def GenreClassifier(input_shape, word_to_vec_map, word_to_index, nbClasses):

    """
	Adapted from Deep Learning Specialization by deeplearning.ai: https://www.coursera.org/specializations/deep-learning?

    Function creating the graph of the model
    
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras 
    """
    
    # Define input of the graph of dtype 'int32' as it contains indices
    sentence_indices = Input(shape = input_shape, dtype = 'int32')
    
    # Create the embedding layer pretrained with GloVe Vectors 
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    # Propagate sentence_indices through your embedding layer, you get back the embeddings
    embeddings = embedding_layer(sentence_indices)
    
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # output is a batch of sequences
    X = LSTM(128, return_sequences = True)(embeddings)
    
    # Add dropout with a probability of 0.5
    X = Dropout(0.2)(X)
    
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128, return_sequences = False, return_state = False)(X)
    
    # Add dropout with a probability of 0.5
    X = Dropout(0.5)(X)
    
    # Propagate X through a Dense layer with softmax activation to get back a batch of 23-dimensional vectors.
    X = Dense(nbClasses)(X)
    
    # Add a softmax activation
    X = Activation('softmax')(X)

    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=sentence_indices, outputs=X)
        
    return model


def convert_to_one_hot(Y, C):

	"""
	Converts the labels to one hot encoding
    
    Arguments:
    Y -- labels
    C -- number of classes

    Returns:
    Y -- one hot encoded labels 
    """

    Y = np.eye(C)[Y.reshape(-1)]
    return Y



def plot_model_history(model_history, fig_name):

	"""
	Plots the history of the model, credits of function to:
	http://parneetk.github.io/blog/neural-networks-in-keras/
    
    Arguments:
    model_history -- history of the model
    fig_name -- name of the figures to be saved 
    """
    
    plt.figure()
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.savefig(fig_name)
    plt.show()


def trainModelV2(X_train_indices, Y_train_oh, word_to_vec_map, word_to_index, max_length, summary = False, 
               dropout_rate = 0.2, batch_size = 32, epochs = 50, loss ='categorical_crossentropy', 
               optimizer ='adam'):

	"""
    Function creating, defining the training conditions of the model and fitting it to our data. 
    Takes into account the weights of the classes.
    
    Arguments:
    X_train_indices -- sentences converted to their respective indices in the word to index dictionnary
    Y_train_oh -- one hot encoding of the labels
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)
    max_length -- maximum length of an overview 

    Returns:
    history -- history of the model
    model -- a model instance in Keras 
    """
	
	# define model 
	model, logits = GenreClassifierV2((max_length,), word_to_vec_map, word_to_index, len(df["genres"].unique()), dropout_rate = dropout_rate)

	if summary:
		# print the summary of the model
		model.summary()

	# determine the weight of each class 
	class_weights = pd.read_csv('datasets/genreLabels.csv')
	class_weights = class_weights['weight'].copy().to_dict()
	
	model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

	# add a checkpoint to check the performance of the model every 5 epochs and save the model if the validation accuracy has improved
	modelcheckVal = ModelCheckpoint('models/validation-weights-improvement-{epoch:02d}-{val_acc:.2f}.h5', monitor='val_acc', period=5, verbose=1, save_best_only=True, mode='max')
	callbacks_list = [modelcheckVal] 

	history = model.fit(X_train_indices, Y_train_oh, epochs = epochs, 
		callbacks=callbacks_list, batch_size = batch_size, validation_split = 0.1, shuffle=True, class_weight = class_weights)

	return history, model


def trainModel(X_train_indices, Y_train_oh, word_to_vec_map, word_to_index, max_length, summary = False, 
               dropout_rate = 0.2, batch_size = 32, epochs = 50, loss ='categorical_crossentropy', 
               optimizer ='adam'):

	"""
    Function creating, defining the training conditions of the model and fitting it to our data. 
    
    Arguments:
    X_train_indices -- sentences converted to their respective indices in the word to index dictionnary
    Y_train_oh -- one hot encoding of the labels
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)
    max_length -- maximum length of an overview 

    Returns:
    history -- history of the model
    model -- a model instance in Keras 
    """
    
    # define model 
    model, logits = GenreClassifierV2((max_length,), word_to_vec_map, word_to_index, len(df["genres"].unique()))
    
    if summary:
    	# print the summary of the model
        model.summary()
        
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    
    # add a checkpoint to check the performance of the model every 5 epochs and save the model if the validation accuracy has improved
    modelcheckVal = ModelCheckpoint('models/validation-weights-improvement-{epoch:02d}-{val_acc:.2f}.h5', monitor='val_acc', period =5, verbose=1, save_best_only=True, mode='max')
    callbacks_list = [modelcheckVal]
    
    history = model.fit(X_train_indices, Y_train_oh, epochs = 50, 
                             callbacks=callbacks_list, batch_size = batch_size, validation_split = 0.1, shuffle=True)

    return history, model



if __name__ == "__main__":
   
	df = pd.read_csv('datasets/preprocessed.csv')
	df.dropna(inplace = True)

	# obtain the GloVe dataset of dimensionality 100
	word_to_index, index_to_word, word_to_vec_map = read_glove_vecs_only_alpha('datasets/glove.6B/glove.6B.100d.txt')

	# determine the maximum length of a movie overview
	max_sequence_length = df["overview length"].max()

	# split the data into training and testing sets
	X = df['overview'].values
	y = df['genre label'].values

	# convert the sentences to their respective indices in the word to index dictionnary
	X_indices = sentences_to_indices(X, word_to_index, max_sequence_length)

	# one-hot encode the labels
	y_oh = convert_to_one_hot(y, C = len(df["genres"].unique()))

	# train the model and keep its history
	history, model = trainModelV2(X_indices, y_oh, word_to_vec_map, word_to_index, max_length = max_sequence_length)

	# generate a plot of the model's progress over time and save the figure
	plot_model_history(history, 'graphs/weighted_model_categorialloss_dropout.png')

	# train the model and keep its history
	history, model = trainModel(X_indices, y_oh, word_to_vec_map, word_to_index, max_length = max_sequence_length)

	# generate a plot of the model's progress over time and save the figure
	plot_model_history(history, 'graphs/model_categorialloss_dropout.png')
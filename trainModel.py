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


def GenreClassifierV2(input_shape, word_to_vec_map, word_to_index, nbClasses):
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
    X = Dropout(0.5)(X)
    
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
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


#credits of function to http://parneetk.github.io/blog/neural-networks-in-keras/
def plot_model_history(model_history, fig_name):
    
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
               dropout_rate = 0.5, batch_size = 32, epochs = 50, loss ='categorical_crossentropy', 
               optimizer ='adam'):

	model, logits = GenreClassifierV2((max_length,), word_to_vec_map, word_to_index, len(df["genres"].unique()))

	if summary:
		model.summary()

	class_weights = pd.read_csv('datasets/genreLabels.csv')
	class_weights = class_weights['weight'].copy().to_dict()

    # your class weights
	#class_weights = tf.constant([class_weights])
	# deduce weights for batch samples based on their true label
	#weights = tf.reduce_sum(class_weights * Y_train_oh, axis=1)
	# compute your (unweighted) softmax cross entropy loss
	#unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(Y_train_oh, logits)
	# apply the weights, relying on broadcasting of the multiplication
	#weighted_losses = unweighted_losses * weights
	# reduce the result to get your final loss
	#loss = tf.reduce_mean(weighted_losses)
	#loss = unweighted_losses
	
	model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

	#earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=3, verbose=1, mode='auto')
	modelcheckVal = ModelCheckpoint('modelsAWS2/validation-weights-improvement-{epoch:02d}-{val_acc:.2f}.h5', monitor='val_acc', period=5, verbose=1, save_best_only=True, mode='max')
	callbacks_list = [modelcheckVal] #, modelcheckTrain]

	history = model.fit(X_train_indices, Y_train_oh, epochs = 50, 
		callbacks=callbacks_list, batch_size = batch_size, validation_split = 0.1, shuffle=True, class_weight = class_weights)

	return history, model


def trainModel(X_train_indices, Y_train_oh, word_to_vec_map, word_to_index, max_length, summary = False, 
               dropout_rate = 0.5, batch_size = 32, epochs = 50, loss ='categorical_crossentropy', 
               optimizer ='adam'):
    
    model, logits = GenreClassifierV2((max_length,), word_to_vec_map, word_to_index, len(df["genres"].unique()))
    
    if summary:
        model.summary()
        
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    
    #earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=3, verbose=1, mode='auto')
    modelcheckVal = ModelCheckpoint('modelsAWS/validation-weights-improvement-{epoch:02d}-{val_acc:.2f}.h5', monitor='val_acc', period =5, verbose=1, save_best_only=True, mode='max')
    callbacks_list = [modelcheckVal] #, modelcheckTrain]
    
    history = model.fit(X_train_indices, Y_train_oh, epochs = 50, 
                             callbacks=callbacks_list, batch_size = batch_size, validation_split = 0.1, shuffle=True)

    return history, model




df = pd.read_csv('datasets/preprocessed.csv')
df.dropna(inplace = True)

# obtain the GloVe dataset of dimensionality 100
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs_only_alpha('datasets/glove.6B/glove.6B.100d.txt')

# determine the maximum length of a movie overview
df['overview length'] = df['overview'].apply(lambda x: len(str(x).split(' ')))
max_sequence_length = df["overview length"].max()

# split the data into training and testing sets
X = df['overview'].values
y = df['genre label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# convert the sentences to their respective indices in the word to index dictionnary
X_train_indices = sentences_to_indices(X_train, word_to_index, max_sequence_length)

# one-hot encode the labels
y_train_oh = convert_to_one_hot(y_train, C = len(df["genres"].unique()))

# train the model and keep its history
history, model = trainModel(X_train_indices, y_train_oh, word_to_vec_map, word_to_index, max_length = max_sequence_length)

# generate a plot of the model's progress over time and save the figure
plot_model_history(history, 'graphs/model_categorialloss.png')

# evaluate the accuracy of the model on the test set
X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = max_sequence_length)
y_test_oh = convert_to_one_hot(y_test, C = len(df["genres"].unique()))
loss, acc = model.evaluate(X_test_indices, y_test_oh)

print("Test accuracy = ", acc)

'''
history, model = trainModel(X_train_indices, y_train_oh, word_to_vec_map, word_to_index, max_length = max_sequence_length)

# save the model
#model.save_weights("models/Epochs50_Adam_CCloss_V2.h5") 

# generate a plot of the model's progress over time and save the figure
plot_model_history(history, 'graphs/model_categorialloss.png')

# evaluate the accuracy of the model on the test set
X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = max_sequence_length)
y_test_oh = convert_to_one_hot(y_test, C = len(df["genres"].unique()))
loss, acc = model.evaluate(X_test_indices, y_test_oh)

print("Test accuracy = ", acc)
'''
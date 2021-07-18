import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Activation, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.models import Model
from keras.utils import to_categorical

def create_dataset_array(df, labelname, labels_present=False):
    '''
    The function takes the dataframe (train/valid/test) as input, processes it for training or testing in a character-level CNN
    '''
    #step1: lower the alphabets
    texts = df["text"].values 
    texts = [s.lower() for s in texts] 

    #step2: build a character dictionary 
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789 ,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    char_dict = {}
    for i, char in enumerate(alphabet):
        char_dict[char] = i + 1


    #step3: convert strings to tokens which are characters
    #Tokenizer
    tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
    tk.fit_on_texts(texts)
    # Use char_dict to replace the tk.word_index
    tk.word_index = char_dict.copy()
    # Add 'UNK' to the vocabulary 
    tk.word_index[tk.oov_token] = max(char_dict.values()) + 1

    #step 4: Convert string to index 
    sequences = tk.texts_to_sequences(texts)

    #step5: Pad and trancate the sequences as instructed by paper
    data = pad_sequences(sequences, maxlen=3616, padding='post', truncating='post')

    #step 6: convert to numpy array
    data = np.array(data)

    #step 7:prepare the labels:
    if labels_present==True:
        df[labelname] = df[labelname].apply(lambda x:1 if x>=1 else 0)
        classes = to_categorical(df[labelname])
    else:
        classes = None

    return data, classes, tk.word_index, sequences


labelname="Fear"
all_soft = pd.read_pickle("/disk2/sadat/PhishingResearch/processed_work/psych_trait_experiments/Files/Soft_labels_combined.pkl")
train, y_train, char_dict, seq = create_dataset_array(all_soft, labelname, labels_present=True)


#################  MODEL CONSTRUCTION ##############

vocab_size = len(char_dict)
## First, we need to represent the characters with one hot encoded representations to build up a good embedding
# There are 69 characters (1 to 69) and the 70th one is 0, that keras uses for padding
embedding_weights = [] #(71, 70)
embedding_weights.append(np.zeros(vocab_size)) # first row is pad

for char, i in char_dict.items(): # from index 1 to 70
    onehot = np.zeros(vocab_size)
    onehot[i-1] = 1
    embedding_weights.append(onehot)
embedding_weights = np.array(embedding_weights)

# parameter 
input_size = 3616
# vocab_size = 69
embedding_size = 69
conv_layers = [[256, 7, 3], 
            [256, 7, 3], 
            [256, 3, -1], 
            [256, 3, -1], 
            [256, 3, -1], 
            [256, 3, 3]]

fully_connected_layers = [1024, 1024]
num_of_classes = 2
dropout_p = 0.5
optimizer = 'adam'
loss = 'categorical_crossentropy'

# Embedding layer Initialization
embedding_layer = Embedding(vocab_size+1, embedding_size+1, input_length=input_size, weights=[embedding_weights])

# Model 

# Input
inputs = Input(shape=(input_size,), name='input', dtype='int64')  # shape=(?, 1014)
# Embedding 
x = embedding_layer(inputs)
# Conv 
for filter_num, filter_size, pooling_size in conv_layers:
    x = Conv1D(filter_num, filter_size)(x) 
    x = Activation('relu')(x)
    if pooling_size != -1:
        x = MaxPooling1D(pool_size=pooling_size)(x) # Final shape=(None, 34, 256)
x = Flatten()(x) # (None, 8704)
# Fully connected layers 
for dense_size in fully_connected_layers:
    x = Dense(dense_size, activation='relu')(x) # dense_size == 1024
    x = Dropout(dropout_p)(x)
# Output Layer
predictions = Dense(num_of_classes, activation='softmax')(x)
# Build model
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy']) # Adam, categorical_crossentropy

############# Training ###########
# Training
model.fit(train, y_train,
        batch_size=16,
        epochs=10,
        verbose=1)


#####  Prediction ##########


def make_pred_and_save(df_path, model, labelname):
    test_data = pd.read_pickle(df_path)
    test, y_test, _, _ = create_dataset_array(test_data, labelname)
    prediction = model.predict(test)
    test_data["CNN_" + labelname] = prediction[:,1]
    test_data.to_pickle(df_path)

make_pred_and_save("/disk2/sadat/PhishingResearch/processed_work/psych_trait_experiments/Files/combined_iwspa_ap_train_set.pkl", model=model, labelname="Fear")
make_pred_and_save("/disk2/sadat/PhishingResearch/processed_work/psych_trait_experiments/Files/iwspa_ap_test.pkl", model=model, labelname="Fear")
make_pred_and_save("/disk2/sadat/PhishingResearch/processed_work/psych_trait_experiments/Files/spamSMS.pkl", model=model, labelname="Fear")
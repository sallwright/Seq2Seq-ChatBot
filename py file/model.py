
# coding: utf-8

# # Creating a ChatBot using TensorFlow with RNN and LSTM
# 
# The assignment: 
# 
# 1. Read the json file and create a proper data structure.
# 2. Shuffle the data and create a 80-10-10 split (80% for training, 10% for development, and 10% for testing). 
# 3. Train a neural net. 
# 4. Create an api (use some Python framework like Flask, and remember to set an API key so that you protect the data).
# 5. Write a Readme.md file with comments on what you have done, how to run the code, and how to deploy on Google Cloud.
# 6. Put everything on github and send me a link
# 
# ### My process for creating the TensorFlow model:
# 1. Import the data from JSON
# 2. Process the data into a suitable format
# 3. Build the model
# 4. Train the model
# 5. Test the model

# In[1]:


import pandas as pd
import numpy as np
import json
import nltk
import tensorflow as tf
import re
import time
import string
from sklearn.model_selection import train_test_split


# ## 1.0 Import the data

# In[2]:


data = json.load(open('data.json'))


# In[3]:


data_main = data['dialogues']


# The file has a lot of metadata, with the data I'm interested in lying in the dialogues section. To make this more intuitive I have gone into this section.

# ## 2.0 Preprocess the data
# The data is currently in a format that is not helpful for my deep learning model. My goals within this sections are to:
# 
# - Create a list of questions whose corresponding answers is in a list of answers.
# - Convert these questions and answers into integers

# In[4]:


# Questions
questions = []

# Answers
answers = []

def sep_data(data):
    for i in range(len(data)):
        for key, value in data[i]['samples'].items():
            for j in range(len(value)):
                questions.append(value[j])
                answers.append(list(data[i]['replies'].values())[0])

sep_data(data_main)

answers = [item for sublist in answers for item in sublist]


# ### 2.1 Cleaning the text
# Here I will clean the text. That is to remove punctuation and convert english word abbreviations. Whilst my data is mostly Norwegian words, there are still English sentences with abbreviations.

# In[5]:


def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm","i am", text)
    text = re.sub(r"he's","he is", text)
    text = re.sub(r"she's","she is", text)
    text = re.sub(r"that's","that is", text)
    text = re.sub(r"what's","what is", text)
    text = re.sub(r"where's","where is", text)
    text = re.sub(r"it's","it is", text)
    text = re.sub(r"\'ll"," will", text)
    text = re.sub(r"\'ve"," have", text)
    text = re.sub(r"\'re"," are", text)
    text = re.sub(r"\'d"," would", text)
    text = re.sub(r"won't","will not", text)
    text = re.sub(r"can't","cannot", text)
    text = re.sub(r'[-()\"#/@;:<>{}+=~|.?,!]','',text)
    return text


# In[6]:


clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question)) 
    
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer)) 


# ### 2.2 Removing words that appear less than 5%
# I will now remove those words that appear less than 5% of the time, this is to improve the accuracy of my model. At the same time I will be creating a dictionary that maps each word to the amount of times that it appears in the data.

# In[7]:


word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1


# In[8]:


threshold = 5
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        questionswords2int[word] = word_number
        word_number += 1
        
answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answerswords2int[word] = word_number
        word_number += 1


# ### 2.2.1 Adding tokens
# I will now add the tokens to this dictionary. These will come in handy later on when we manipulate our strings further for the model.

# In[9]:


tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']

for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
    
for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1


# In[ ]:


# Flipping my answers dictionary from word - integer to integer - word

answersints2word = {w_i: w for w, w_i in answerswords2int.items()}


# In[11]:


# Adding EOS at the end of each answer

for i in range(len(clean_answers)):
    clean_answers[i] += " <EOS>"


# ### 2.3 Translating the words to integers
# As stated at the beginning of this section I need to translate all of my words into integers that correspond to that word.

# In[12]:


questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)
answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)


# ### 2.3.1 Sorting and clipping
# Now that I have my questions and answers in integer format, I will clip those that have a length over 25. Again this is to improve the accuracy of my model, as those sentences over 25 characters will be anomalies

# In[13]:


sorted_clean_questions = []
sorted_clean_answers = []

for length in range(1, 25 + 1):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])


# ## 3.0 Building the model
# The pre-processing of the data is now over, and I have a good structure to work with. Now I need to create a model to take this data and learn from it.
# 
# My process for creating this model will be:
# 1. Define model inputs
# 2. Target preprocessing function
# 3. Create the Encoder RNN layer
# 4. Create Decoder RNN layer 
# 4. Define the Seq2Seq model

# ### 3.1 Model inputs function
# I will begin here with the simple task of defining the placeholders for my model.

# In[15]:


def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob


# ### 3.2 Preprocess targets
# This function will process my targets in a format for my model to understand, including by adding SOS (Start of sentence) at the beginning of each sentence.

# In[16]:


# Will convert the target into the right format: with SOS at the beginning and in two batches
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets


# ### 3.3 Creating the Encoder RNN layer

# In[17]:


def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                                    cell_bw = encoder_cell,
                                                                    sequence_length = sequence_length,
                                                                    inputs = rnn_inputs,
                                                                    dtype = tf.float32)
    return encoder_state


# ### 3.4 Create the Decoder RNN Layer 
# This part of the model is fairly involved as I need to split this section into 3 parts to create my decoder RNN layer:
# 1. Decode the training set
# 2. Decode the validation set
# 3. Decode the RNN Layer

# In[19]:


# Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)


# In[20]:


# Decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions


# In[21]:


# Create the Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions


# ### 3.5 Defining the seq2seq model

# In[22]:


def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions


# ## 4.0 Training the model
# Our RNN model has been built, now it's time to train the model on my data.
# 
# My process for training the model will be as follows:
# 1. Initial set-up
# 2. Define loss error, optimizer etc.
# 3. Further string manipulation
# 4. Training for loop

# ### 4.1 Initial set-up
# To set up my model training I want to achieve the following:
# - Define my hyperparameters
# - Define my session
# - Load the model inputs
# - Set length of sequence and the shape of the inputs
# - Get my training and test predictions

# ### 4.1.1 Setting hyperparameters
# Training a deep learning model takes time. I had to make a pay-off between optimal chatbot performance and training time. To strike a good balance I decided on the following parameters which resulted in a training time of roughly 3 hours. If I had more time then this model could be even more accurate by using better parameters and training within the cloud.

# In[23]:


epochs = 10
batch_size = 128
rnn_size = 512
num_layers = 2
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate= 0.0001
keep_probability = 0.5


# In[24]:


# Define my session
tf.reset_default_graph()
session = tf.InteractiveSession()


# In[25]:


# Loading model inputs
inputs, targets, lr, keep_prob = model_inputs()


# In[26]:


# Setting sequence length
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')


# In[ ]:


# Setting the shape of inputs tensor
input_shape = tf.shape(inputs)


# In[28]:


# Getting training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerswords2int),
                                                       len(questionswords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionswords2int)


# ### 4.2 Defining loss error, optimizer etc.

# In[29]:


with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)


# ### 4.3 Further string manipulation
# Before I enter my training for loop I need to do some further string manipulation. Namely:
# 
# - Add padding so each sentence is the same length
# - Split my data into the batches that will be fed into the model
# - Train, dev, test split

# In[ ]:


# Padding the sequences with PAD
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]


# In[31]:


# Split the data into batches of questions and answers
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int))
        yield padded_questions_in_batch, padded_answers_in_batch


# ### 4.3.1 Train, dev, test split
# One of my specific tasks for this project was to create a train, dev, test split that was 80,10,10. To do this I first created a train, test split using SkLearn (which also shuffled my data as part of the process) and then split that training data into train and development data. 
# 
# I am then left with three groups of data:
# 1. Training Questions, Answers
# 2. Validation Questions, Answers
# 3. Questions Shuffled Testing, Answers Shuffled Testing
# 
# The testing data can then be used later to test the model.

# In[32]:


# Splitting questions and answers into training, validation, and test sets
questions_shuffled, questions_shuffled_testing, answers_shuffled, answers_shuffled_testing = train_test_split(sorted_clean_questions,sorted_clean_answers,test_size=0.1,shuffle=True,random_state=101)

# questions_shuffled, answers_shuffled - Will now split these into the testing and validation

training_validation_split = int(len(questions_shuffled) * 0.1)
training_questions = questions_shuffled[training_validation_split:]
training_answers = answers_shuffled[training_validation_split:]
validation_questions = questions_shuffled[:training_validation_split]
validation_answers = answers_shuffled[:training_validation_split]


# ### 4.4 Training for loop
# This is the section where the model will learn my data.

# In[33]:


# Training
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 100
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
print("Game Over")


# ## 5.0 Testing my chatbot
# My model has now run through the training and is ready to be implemented. This is a short section and will go through this process:
# 
# 1. Begin the session
# 2. Convert question to integers
# 3. Setting up the chat

# In[37]:


# Loading the weights and Running the session
checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)


# In[38]:


# Converting the questions to integers
def convert_string2int(question, word2int):
    question = clean_text(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]


# In[ ]:


# Setting up the chat
while(True):
    question = input("You: ")
    if question == 'Goodbye':
        break
    question = convert_string2int(question, questionswords2int)
    question = question + [questionswords2int['<PAD>']] * (25 - len(question))
    fake_batch = np.zeros((batch_size, 25))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if answersints2word[i] == 'i':
            token = ' I'
        elif answersints2word[i] == '<EOS>':
            token = '.'
        elif answersints2word[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answersints2word[i]
        answer += token
        if token == '.':
            break
    print('ChatBot: ' + answer)


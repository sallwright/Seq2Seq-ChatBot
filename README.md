# Building a chatbot using TensorFlow
I was tasked with create a chatbot for some data within a JSON file. The sub-tasks for this project were:
1. Read JSON file
2. Shuffle data and create a train, development, test split
3. Build and train the neural net
4. Create API, probably using Flask
5. Describe how to run the code and how to deploy to Google Cloud

## How the model works
To create this chatbot model I decided to use a Sequence to Sequence (seq2seq) model with RNN and LSTM cells.

The seq2seq model has two RNN's, an encoder and a decoder. The encoder reads the input, word by word, and emits a context which is the essence of the input. Based on this, the decoder generates the output, one word at a time while looking at the context and the previous word during each timestep.

![RNN Diagram](https://github.com/sallwright/TensorFlow_ChatBot/blob/master/img/RNN%20Diagram.png)

My model uses LSTM cells. An LSTM cell is the building block of my RNN layers and is loosely based on the neurons one would find in a brain. It's main role (in a very simple description) is to remember values over certain arbitrary time intervals.

![LSTM Diagram](https://github.com/sallwright/TensorFlow_ChatBot/blob/master/img/LSTM%20Diagram.png)

## How to run the code
Download model.ipynb or model.py along with the weights files and run within the command line or an iPython notebook.

## How to deploy on Google Cloud
Once I have written the Flask app for this chatbot model, then I would be able to deploy this to Google Cloud.

This would be a two step process (assuming that I have all the prerequisites locally).

1. run command:  'gcloud app deploy'


This would build a container image and then deploy it to the App Engine.

2. run command: 'gcloud app browse'

Which will launch the browser to view the app

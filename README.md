This project was developed for a NLP class under supervision of Dr Ta.

We use the BERT embedding and a Pytorch LSTM model (BERT => LSTM => LINEAR) to build our classification system.

Our input consists in a dataset of messages (being Questions, Comments or Answers) based on various forums in different languages. The target consists in identifying wether the message is the best answer.

We first transform our CSV into a JSON file that is easier to work with.

For the sake of simplicity we will be using stochastic training while working with our json data.

In order to use BERT embedding we use the FLAIR project that is based on pytorch and return Torch Tensors. We feed these tensors into an LSTM layer that then feeds its outputs to a regular Dense layer with a sigmoid function. 

We build our input by encapsulating information into particular [TAGS]. We concat the tagged message with tagged metadata and use Flair to embed the input.

You can train the model launching the runQnA.py file with the arguments you wish to use. You also use this file to predict whatever message you wish classifying. You can also train the model on google colab provided you feed it with a json in the right format.

Further work could be done by creating a non standard model that uses mini-batch and two individual network: one for the LSTM and one for the meta-data that we combine to one final FC layer to output our result.
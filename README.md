This project was developed for a NLP class under supervision of Dr Ta.

We use the BERT embedding and a Pytorch LSTM model (BERT => LSTM => LINEAR) to build our classification system.

Our input consists in a dataset of messages (being Questions, Comments or Answers) based on various forums in different languages. The target consists in identifying whether the message is the best answer or not.

We first transform our CSV into a JSON file that is easier to work with.

For the sake of simplicity we will be using stochastic training while working with our json data.

In order to use BERT embedding we use the FLAIR project that is based on pytorch and return Torch Tensors. We feed these tensors into an LSTM layer that then feeds its outputs to a regular Dense layer with a sigmoid function. 

On the other hand we also feed the meta-data to a regular dense network. We then use both outputs on a final layer that optimizes the decision and finally predicts a binary.

We build our input by encapsulating information into particular [TAGS]. We concat the tagged message with tagged title and use Flair to embed the input. We also regularize the datatype of our meta-data and directly send it in a Dense network.

![alt text](https://github.com/MCSZN/qna/blob/master/imgs/model.png)

You can train the model launching the runQnA.py file with the arguments you wish to use. You also use this file to predict whatever message you wish classifying. You can also train the model on google colab provided you feed it with a json in the right format.

Further work would be to optimize the number of LSTM and Dense layers. We could also check which BERT version is most appropriate for this particular task given that the language is majoritarily english. Once these improvements are done we could also optimize the actual architecture trying out if CNN might be useful for the meta-data.
A final improvement would be to use regularizers as well as normalizers to even further accelerate learning with learning rate decay to avoid overfitting and even residual networks to allows us creating deeper networks. 

To use this repo you can directly download it, launch the "setup.py" file with your python interpreter and eventually use the "runQnA.py" file to directly train it with the parameters you like!

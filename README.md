# Venture Funding, DeepLearning Model

The following is a Deep Learning model that aims to predicts whether funding applicants will become successful based on specific features.  
#### Data

The dataset contains approximately 34,000 organisations that have received funding over the years. This dataset contains various types of information about the organisations, including whether they ultimately became successful.

The provided dataset was used to create a binary classifier model.

Here are the fields in the dataset:

* EIN
* NAME
* APPLICATION_TYPE
* AFFILIATION
* CLASSIFICATION
* USE_CASE
* ORGANIZATION
* STATUS
* INCOME_AMT
* SPECIAL_CONSIDERATIONS
* ASK_AMT
* IS_SUCCESSFUL

#### Data Transformation

As a start EIN (Employer ID Number) and NAME columns were dropped from the dataframe as these would not impact the outcome of IS_SUCCESSFUL.  

Next `OnHotEncoder` was used to encode the categorical variables:

* 'APPLICATION_TYPE',
* 'AFFILIATION',
* 'CLASSIFICATION',
* 'USE_CASE',
* 'ORGANIZATION',
* 'INCOME_AMT',
* 'SPECIAL_CONSIDERATIONS'

The numerical and encoded data were combined to create a numerical dataframe for use in the model.

`train_test_split` function was used to split the encoded data into X_train, X_test, y_train, y_test and X_train and X_test were subsequently scaled using `StandardScaler` function.

#### Binary Classification Model (Deep Neural Network using Tensorflow Keras)

A two layer deep neural network model using `relu` acivation function for both layers was created.  The ouput layer activation function `sigmoid` was used for binary output.  

![Oringal Deep Learning Model](https://github.com/KSohi-max/Venture_Fund_DeepLearning/blob/main/Images/Original_DL_Model.png)

The model was compiled and fitted using `binary_crossentropy` loss function and optimized using `adam` optimizer.  The model was evaluated on `accuracy`.  The model was fitted using 50 epoches and the training data.

![Model Accuracy](https://github.com/KSohi-max/Venture_Fund_DeepLearning/blob/main/Images/Original_DL_Model%20Accuracy.png)

The model achieved a 0.7289795875549316 accuracy.
The model results and weights were saved under [`conAlphabetSoup.h5`](https://github.com/KSohi-max/Venture_Fund_DeepLearning/blob/main/conAlphabetSoup.h5). 

#### Binary Classification Model_A1

In the second model, the number of neurons were increase, a third hidden layer was added.  Also the activation function of all hidden layers were changed to `LeadyRelu` (note the additional layer required to use LeakyRelu activation function). The ouput layer activation function remain as `sigmoid`. 

![Deep Learning Model_A1](https://github.com/KSohi-max/Venture_Fund_DeepLearning/blob/main/Images/DL_Model_A1.png)

The model A1 was also compiled and fitted using `binary_crossentropy` loss function and optimized using `adam` optimizer.  The model was evaluated on `accuracy`.  The model was fitted using 50 epoches and the training data.

![Model_A1 Accuracy](https://github.com/KSohi-max/Venture_Fund_DeepLearning/blob/main/Images/DL_Model_A1%20Acurracy.png)

Model_A1 achieved an accuracy of approximately 0.7301457524299622. 
The model results and weights were saved under [`AlphabetSoup_A1.h5`](https://github.com/KSohi-max/Venture_Fund_DeepLearning/blob/main/AlphabetSoup_A1.h5). 

#### Binary Classification Model_A2

In the third and final model, the number of neurons in hidden layer were the same as Model_A1 (i.e. increased compared to Orginal Model), the number of hidden layers was decreased back to two layers, similar to the Original Model.  But the activation functions were changed to `Softmax` for the hidden layers and for ouput layer to `tanh`.

![Deep Learning Model_A2](https://github.com/KSohi-max/Venture_Fund_DeepLearning/blob/main/Images/DL_Model_A2.png)

The model A2 was also compiled and fitted using `binary_crossentropy` loss function and optimized using `adam` optimizer.  The model was evaluated on `accuracy`.  The model was fitted using 100 epoches, increased number of iterations, and the training data.

![Model_A2 Accuracy](https://github.com/KSohi-max/Venture_Fund_DeepLearning/blob/main/Images/DL_Model_A2%20Acurracy.png)

Model_A1 achieved an accuracy of approximately 0.7297959327697754. 
The model results and weights were saved under [`AlphabetSoup_A2.h5`](https://github.com/KSohi-max/Venture_Fund_DeepLearning/blob/main/AlphabetSoup_A2.h5).

#### Discussion

In reviewing the results,i.e., improvement in accuracy, one can see a VERY slight improvement in `accuracy` from Model_A2 while the Original Model and Model_A2 results show about the same `accuracy`.  

This may indicate that more nuerons and additional hidden layers are likely to help the model improve in `accuracy`.  Also note that change in the activation function from Relu to LeakyRelu may have contributed to the slight improvement in the results.  

Note the difference betwee `relu` and `LeakyRelu` functions:

The ReLU Activation Function helped alleviate the [saturation](https://towardsdatascience.com/leaky-relu-vs-relu-activation-functions-which-is-better-1a1533d0a89f) occurring when a model utilized the Sigmoid Activation Function. It issue in two ways.

1. It is not constrained by zero and one like the Sigmoid Activation Function (along the y-axis).
2. Any value that is negative will be sent to zero.

The Leaky ReLU Activation Function (LReLU) is very similar to the ReLU Activation Function with one change. Instead of sending negative values to zero, a very small slope parameter is used which incorporates some information from negative values.

Note: "What is saturation? It aligns with the exploding and vanishing gradients problem which arises when training a neural network. When gradients “explode”, the activations are sent to extremely large numbers and the update in model weights is too big, leading to a model that can not learn to solve a given task. When gradients “vanish,” the updates in the model’s weights become so small that they are unable to be changed to conform and solve the given problem." (Source:  https://towardsdatascience.com/leaky-relu-vs-relu-activation-functions-which-is-better-1a1533d0a89f)

It is important to continue to work with the model to determine if it can be improved by either adding additional features in the dataset, understanding of important features that impact the accuracy of prediction, and more cases of both successful and unsuccessful applicants, in other words, more data.

#### Conclusion

An `accuracy` of 73% is valid although a higher accuracy is preferred for a model that would be implented in practice.  In the case of VC funding aiming to predict a successful company for investment, this level of accuracy may be sufficient for preliminary review of applicants seeking funding.  

It is however, recommended that: 

* VC use the model or variation of the model above as a first screening of applicants only.
* VC should further conduct deep due-diligence into the companies that are predicted to be successful by the above model(s).
* VC should conduct face-to-face review of the applicants to understand what improvements are required on many aspects of the company (legal, financial, organizational, physical, virtual (online+tech), etc.) before making any final decisions to invest funds.

# Review Classification
This repository contains code, data and report for review classification.

## The API package is contained in the API folder
It contains three mean functions: trainCV(), train(). and predict(). Simply import main.py and call these functions.

## The usage of these functions
train_CV(train_file, category, result_file, k_fold=5):
- function for training the model using cross validation. 
- train_file is the input training data path (.xlsx file)
- category is 'context', 'content', or 'driver' which specifies the classification task.
- result_file is the output file path (.csv file) for saving the prediction results.
- k_fold is the number of fold for cross validation. Default is 5.

train(train_file, category):
- function for training the model using whole training data.
- train_file is the input training data path (.xlsx file)
- category is 'context', 'content', or 'driver' which specifies the classification task.
- the function returns the trained cnn_model

predict(test_file, cnn_model, result_file):
- function for predicting the test data using the trained model
- test_file is the input testing data path (.xlsx file)
- cnn_model is the trained CNN model returned from function 'train()' 
- result_file is the output file path (.csv file) for saving the prediction results.


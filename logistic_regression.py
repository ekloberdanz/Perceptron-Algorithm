import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

# Import dataset
data = pd.read_csv("spambase.data")
data.columns = [i for i in range(0, data.shape[1])] 

# Divide data into train and test set
spam = data[data[57] == 1]
ham = data[data[57] == 0]

number_of_spam_train = int(spam.shape[0] * (2/3))
number_of_ham_train = int(ham.shape[0] * (2/3))

spam_train = spam.iloc[0: number_of_spam_train, :]
spam_test = spam.iloc[number_of_spam_train: spam.shape[0], :]

ham_train = ham.iloc[0: number_of_ham_train, :]
ham_test = ham.iloc[number_of_ham_train: ham.shape[0], :]

# Create training set that contains the first 2/3 of spam and first 2/3 of ham
train_set = spam_train.append(ham_train)
x_train = train_set.iloc[:, 0:57]
y_train = train_set.iloc[:, -1]

# Create testing set that contains the last 1/3 of spam and last 1/3 of ham
test_set = spam_test.append(ham_test)
x_test = test_set.iloc[:, 0:57]
y_test = test_set.iloc[:, -1]


# Sigmoid function
def sigmoid(z):
	return(1/(1 + np.exp(-z)))

# Prediction
def predict(input_data, theta, threshold):
	theta_transpose = np.transpose(theta)
	predictions = sigmoid(np.dot(input_data, theta_transpose))
	classifications = []
	for prediction in predictions:
		if prediction >= threshold:
			classifications.append(1) # spam
		else:
			classifications.append(0) # ham
	return classifications

# Function that learns the parameters
def learn_theta(X, y, learning_rate, number_of_epochs, convergence_threshold, threshold):
	theta = np.zeros(X.shape[1]) # Initialize theta to zeros
	prediction = predict(x_train, theta, threshold)
	accuracy = np.sum(np.equal(prediction, y_train))/ y_train.size
	#while accuracy < convergence_threshold:
	# also possible to iterate over number of epochs: 
	for i in range(number_of_epochs):
		z = np.dot(X, theta)
		h = sigmoid(z)
		gradient = np.dot(X.T, (h-y)) / y.size
		theta -= learning_rate * gradient
		prediction = predict(x_train, theta, threshold)
		accuracy = np.sum(np.equal(prediction, y_train))/ y_train.size
	return(theta)

# Function for experimenting with learning rate
def experiment(learning_rate, norm, number_of_epochs):
	if norm == True:
		learned_theta = learn_theta(normalize(x_train, axis=1), y_train, learning_rate, number_of_epochs, 0.7, 0.5)
		prediction = predict(normalize(x_test), learned_theta, 0.5)
	else:
		learned_theta = learn_theta(x_train, y_train, learning_rate, number_of_epochs, 0.7, 0.5)
		prediction = predict(x_test, learned_theta, 0.5)
	test_accuracy = 100 * (np.sum(np.equal(prediction, y_test))/ y_test.size)
	print("The accuracy on test data of logistic regression model with learning rate = ",learning_rate, "and number of epochs = ", number_of_epochs, "is: ", test_accuracy, "%")
	return(test_accuracy)

# Test accuracies of models with different learning rates
learning_rates = [0.01, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 0.99]
number_of_epochs = [0, 10, 20, 50, 80, 100, 150, 200, 300]

print("\nUsing un-normalized data")
lr_un_norm_accuracy = [experiment(lr, False, 100) for lr in learning_rates]
epoch_un_norm_accuracy = [experiment(0.2, False, epoch) for epoch in number_of_epochs]

print("\nUsing normalized data")
lr_norm_accuracy = [experiment(lr, True, 100) for lr in learning_rates]
epoch_norm_accuracy = [experiment(0.2, True, epoch) for epoch in number_of_epochs]

# Plot learning rates and accuracies
plt.plot(learning_rates, lr_un_norm_accuracy, '-', color='g', label='un-normalized data')
plt.plot(learning_rates, lr_norm_accuracy, '-', color='b', label='normalized data')
plt.title("Experiments with learning rate")
plt.xlabel("learning rate")
plt.ylabel("accuracy on test data in %")
plt.legend(loc='lower left')
plt.show()

# Plot epochs and accuracies
plt.plot(number_of_epochs, epoch_un_norm_accuracy, '-', color='g', label='un-normalized data')
plt.plot(number_of_epochs, epoch_norm_accuracy, '-', color='b', label='normalized data')
plt.title("Experiments with number of epochs")
plt.xlabel("number of epochs")
plt.ylabel("accuracy on test data in %")
plt.legend(loc='lower left')
plt.show()


# check that total number of examples is correct
#print(spam_train.shape[0] + spam_test.shape[0] + ham_train.shape[0] + ham_test.shape[0])

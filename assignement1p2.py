import numpy as np
import matplotlib.pyplot as plt

data = np.array([
	[1, 2, 1], 
	[1, 4, 1], 
	[2, 2, 1], 
	[4, 2, -1], 
	[3, 4, -1], 
	[2, 3, -1]
	])

X = data[:, 0:2]
Y = data[:, -1]


def predict(input_data, theta):
	theta_transpose = np.transpose(theta)
	perceptron = np.dot(theta_transpose, input_data)
	if perceptron >= 0:
		return (1)
	else:
		return(-1)

# Function that learns the bias and weights
def learn_theta(data, learning_rate):
	theta = np.zeros(3) # Initialize theta [bias, weight1, weight2] = [0, 0, 0]
	prediction = [predict(np.append([1], x), theta) for x in X]
	# Repeat this procedure until the entire training set is classified correctly
	# Alternatively, if not linearly separable, to avoid infinite loop I can do for i in range(number_of_epochs):
	while np.all(prediction == Y) != True:
		# For each example
		for i in data:
			y = i[-1]
			x = np.append([1], i[0:2]) # add 1 at the begining of each x vector
			pred = predict(x, theta) # predict y'
			# if mistake update parameters
			if pred != y:
				theta += learning_rate * np.dot((y-pred), x)
				pred = predict(x, theta)
		prediction = [predict(np.append([1], x), theta) for x in X]
	return(theta)

def plot_hyperplane(X, y, weights, bias):
    """
    Plots the dataset and the estimated decision hyperplane
    """
    slope = - weights[0]/weights[1]
    intercept = - bias/weights[1]
    x_hyperplane = np.linspace(1,5,2)
    y_hyperplane = slope * x_hyperplane + intercept
    fig = plt.figure(figsize=(10,10))
    # plot points
    for i in data:
	    if i[2] >= 1:
		    plt.scatter(i[0], i[1], marker='+', s=100)
	    else:
	        plt.scatter(i[0], i[1], marker='_', s=100)
	# plot decision boundary
    plt.plot(x_hyperplane, y_hyperplane, '-', color='g')
    plt.title("Dataset and fitted decision hyperplane")
    plt.xlabel("x_1")
    plt.ylabel("x_2")
    plt.ylim(1, 5)
    plt.show()

# Learned bias and weights
learned_theta = learn_theta(data, learning_rate=0.1)
print("Learned theta: [bias, w_1, w_2] = ", learned_theta)

# Predictions
prediction = [predict(np.append([1], x), learned_theta) for x in X]
print("Predicted labels: ", prediction)
print("Actual labels: ", Y)

# Plot with data and final hyperplane
plot_hyperplane(X, Y, learned_theta[1:3], learned_theta[0])
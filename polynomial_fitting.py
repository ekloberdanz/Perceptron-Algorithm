# Import libraries
import random
import numpy as np
import math
import matplotlib.pyplot as plt

# Generate Y matrix
def Y_matrix():
	Y = []
	for i in x:
		y = 1 + 2 * math.sin(5*i) - math.sin(15*i)
		Y.append(y) 
	Y = np.array(Y)
	return(Y)

# Generate matrix with residual errors
def E_matrix():
	E = [] 
	for i in x: 
		# Gaussian noise with mean = 0 and unit variance
		noise = float(np.random.normal(0,1,1))
		E.append(noise)
	E = np.array(E)
	return(E)

# Create X matrix
def X_matrix(polynomial_order):
	X = []
	for i in x:
		x_i = [1]
		for k in range(1, polynomial_order+1):
			#print(k)
			x_i.append(np.power(i, k))
		X.append(x_i)
	X = np.array(X)
	return(X)

# Create Beta matrix
def Beta_matrix(X, Y):
	X_transpose = X.T # X transpose
	inverse = np.linalg.inv(X_transpose.dot(X)) # inverse of (X transpose * X)
	Beta = inverse.dot(X_transpose).dot(Y) # # inverse of (X transpose * X) * X transpose * Y
	return(Beta)

# Fit a polynomial of order k
def fit_polynomial(k, Y):
	X = X_matrix(k)
	#print(X)
	B = Beta_matrix(X, Y)
	#print(B.shape)
	fit = X.dot(B)
	return(fit)


def plot_polynomial(poly_order, number_of_times):
	plt.plot(x, f_x, color='r', label='true function') # true function plot
	i=0
	while i < number_of_times:
		error = E_matrix()
		Y = f_x + error # generate y_i for x_i (part b)
		# print data onto graph just once, so it's legible
		if i == 0:
			plt.scatter(x, Y, s=10) # data
		plt.plot(x, fit_polynomial(poly_order, Y), color='black')
		i+=1
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('Polynomial function of order ' + str(poly_order))
	plt.legend(loc='lower left')
	plt.show()


# Generate 51 equally spaced x values between 0 and 1 (part a)
x = list(np.linspace(0,1,51))

 # True function
f_x = Y_matrix()

# Polynomial orders
k = [1, 3, 5, 7, 9, 11]


# Fit a polynomial of order k 30 times for each k
for i in range(len(k)):
	poly_order = k[i]
	plot_polynomial(poly_order, 30)
	i += 1
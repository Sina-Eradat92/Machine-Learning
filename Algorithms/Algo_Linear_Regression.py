# Understanding Linear Regression
# y = mx+b
# m is the slope and b is the y intercept
# m = mean(x) * mean(y) - mean(xy) / mean(x)^2 - mean(x^2)
# b = mean(y) - m * mean(x)
# R^2 (coefficient of determination) = 1 - (SE(y hat) / SE(mean(y)))
# SE = error squared used to penalized outlier sum(yhat - y)^2


from statistics import mean 
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np 
import random

#xs = np.array([ 1,2,3,4,5,6], dtype=np.float64)
#ys = np.array([5,4,6,5,6,7,], dtype=np.float64)

#by adj the variance in this function you can check the acuracy of R^2 
def create_dataset(hm, variance, step=2, correlation=False):
	val = 1
	ys = []
	for i in range(hm):
		y = val + random.randrange(-variance, variance)
		ys.append(y)
		if correlation and correlation == 'pos':
			val+=step
		elif correlation and correlation == 'neg':
			val-=step
	xs = [i for i in range(len(ys))] 
	
	return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_slope(xs,ys):
	m = ( ((mean(xs) * mean(ys)) - mean(xs*ys)) /
		  ((mean(xs)**2) - mean(xs**2)))
	return m
	
def y_intercept(m,ys, xs):
	b = mean(ys) - m * mean(xs)
	return b

def best_fit_slope_and_intercept(xs,ys):
	m = ( ((mean(xs) * mean(ys)) - mean(xs*ys)) /
		  ((mean(xs)**2) - mean(xs**2)))
	b = mean(ys) - m * mean(xs)
	return m, b
	
def squared_error(ys_orig, ys_line):
	return sum((ys_line - ys_orig)**2)
	
def coefficient_of_determination(ys_orig, ys_line):
	y_mean_line = [mean(ys_orig) for y in ys_orig]
	squared_error_reg = squared_error(ys_orig,ys_line)
	squared_error_y_mean = squared_error(ys_orig,y_mean_line)
	return 1 - (squared_error_reg / squared_error_y_mean)

#best fit line 
xs, ys = create_dataset(40,40,2,'pos')
m,b = best_fit_slope_and_intercept(xs,ys)
regression_line = [(m*x)+b for x in xs]
r_squared = coefficient_of_determination(ys,regression_line)

#predictions
predict_x = 8
predict_y = (m*predict_x)+b

#print out
print('m,b: ',m,b)
print('R^2: ',r_squared)

#make and show the modle 
style.use('fivethirtyeight')
plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y, color='g')
plt.plot(xs, regression_line)
plt.show()


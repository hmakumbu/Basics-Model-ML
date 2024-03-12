import numpy as np
import matplotlib.pyplot as plt
from Linear_regression import LinearRegression
from Logestic_regression import LogisticRegression

np.random.seed(0)

# generate data
def generate_data(n= 1000):
  np.random.seed(0)
  x = np.linspace(-5.0, 5.0, n).reshape(-1,1)
  y= (29 * x + 30 * np.random.rand(n,1)).squeeze()
  x = np.hstack((np.ones_like(x), x))
  return x,y

# generate data
x,y= generate_data()
# check the shape
print ((x.shape,y.shape))

def split_data(x,y,train_perc=0.8):
  N=x.shape[0]
  train_size=round(train_perc*N)
  x_train,y_train=x[:train_size,:],y[:train_size]
  x_test,y_test=x[train_size:,:],y[train_size:]
  return x_train,y_train,x_test,y_test

x_train,y_train,x_test,y_test=split_data(x,y)
print(f"x_train:{x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}")


print("------------------ FIRST MODEL LINEAR REGRESSION ---------------------------")

print("------------------First Model Using Batch gradient descent---------------------------")

# First linear class for batch
model1 = LinearRegression(lr=0.01, n_epochs=1000)

# using batch gradient descent
model1.fit_batch_gradient_descent(x_train, y_train)

# Prediction
y_pred_train = model1.predict(x_train)
y_pred_test = model1.predict(x_test)

# Accuracy
accuracy_train = model1.accuracy(y_train, y_pred_train)
accuracy_test = model1.accuracy(y_test, y_pred_test)

print("Training accuracy:", accuracy_train)
print("Test accuracy:", accuracy_test)

#Plot loss
#model1.plot_loss()

print("------------------Second Model Stochastic gradient descent---------------------------")

# Second linear class for batch
model2 = LinearRegression(lr=0.01, n_epochs=1000)

# using stochastic gradient descent
model2.fit_stochastic_gradient_descent(x_train, y_train)

# Prediction
y_pred_train = model2.predict(x_train)
y_pred_test = model2.predict(x_test)

# Accuracu
accuracy_train = model2.accuracy(y_train, y_pred_train)
accuracy_test = model2.accuracy(y_test, y_pred_test)

print("Training accuracy:", accuracy_train)
print("Testing accuracy:", accuracy_test)

#model2.plot_loss()

print("------------------Third Model Stochastic gradient descent---------------------------")

# Second linear class for batch
model3 = LinearRegression(lr=0.01, n_epochs=1000)

# Using mini-batch gradient descent
model3.fit_mini_batch_gradient_descent(x_train, y_train)

# Prediction
y_pred_train = model3.predict(x_train)
y_pred_test = model3.predict(x_test)

# Accuracy
accuracy_train = model1.accuracy(y_train, y_pred_train)
accuracy_test = model1.accuracy(y_test, y_pred_test)

print("Training accuracy:", accuracy_train)
print("Test accuracy:", accuracy_test)

#Plot loss
#model3.plot_loss()

print("------------------ SECOND MODEL LOGISTIQUE REGRESSION ---------------------------")

print("------------------First Model Using fit_mini_batch_gradient_descent---------------------------")

from sklearn.datasets import make_classification
X_class, y_class = make_classification(n_samples= 100, n_features=2, n_redundant=0,
                           n_informative=2, random_state=1,
                           n_clusters_per_class=1)

def split_data_1(x,y,train_perc=0.8):
  N=x.shape[0]
  idx=np.random.permutation(x.shape[0])
  x,y=x[idx],y[idx]
  train_size=round(train_perc*N)
  x_train,y_train=x[:train_size,:],y[:train_size]
  x_test,y_test=x[train_size:,:],y[train_size:]
  return x_train,y_train,x_test,y_test

x_train,y_train,x_test,y_test = split_data_1(X_class,y_class)
print(f"x_train:{x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}")

# Model LogisticRegression for using mini-lots
model = LogisticRegression(lr=0.01, n_epochs=1000, batch_size=32)
model.fit_mini_batch_gradient_descent(x_train, y_train)

# Prediction
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

# Accuracy
accuracy_train = model.accuracy(y_train, y_pred_train)
accuracy_test = model.accuracy(y_test, y_pred_test)

print("Training accuracy:", accuracy_train)
print("Test accuracy:", accuracy_test)

#model.plot_loss()

print("------------------Second Model Using stochastic gradient descent---------------------------")

# Second Model Using fit_stochastic_gradient_descent(
model2 = LogisticRegression(lr=0.01, n_epochs=1000, batch_size=32)
model2.fit_stochastic_gradient_descent(x_train, y_train)

# Prediction
y_pred_train = model2.predict(x_train)
y_pred_test = model2.predict(x_test)

# Accuracy
accuracy_train = model2.accuracy(y_train, y_pred_train)
accuracy_test = model2.accuracy(y_test, y_pred_test)

print("Training accuracy:", accuracy_train)
print("Test accuracy:", accuracy_test)

#model2.plot_loss()

print("------------------Third Model Using stochastic gradient descent---------------------------")

# Third Model Using fit_stochastic_gradient_descent(
model3 = LogisticRegression(lr=0.01, n_epochs=1000, batch_size=32)
model3.fit_batch_gradient_descent(x_train, y_train)


# Prediction
y_pred_train = model3.predict(x_train)
y_pred_test = model3.predict(x_test)

# Accuracy
accuracy_train = model3.accuracy(y_train, y_pred_train)
accuracy_test = model3.accuracy(y_test, y_pred_test)

print("Training accuracy:", accuracy_train)
print("Test accuracy:", accuracy_test)

model3.plot_loss()
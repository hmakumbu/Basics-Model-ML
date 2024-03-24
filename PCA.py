import numpy as np

class PCA:

  def __init__(self, n_component):
    self.n_component = n_component

  def mean(self,X): # np.mean(X, axis = 0)

  # Your code here
    mean = np.sum(X, axis=0)/X.shape[0]
    return mean

  def std(self,X): # np.std(X, axis = 0)

  # Your code here
    std = ((1/(X.shape[0]-1))*np.sum((X-mean(X))**2, axis=0))**0.5

    return std

  def standardize_data(self,X):

  # Your code here
    X_std = (X - mean(X))/std(X)
    return X_std

  def covariance(self,X):

    cov = (1/(X.shape[0]-1))*X.T@X
    return cov

      # compute eigvalues and eigvectors
  def eigV(self,X) :

     Cov_mat = self.covariance(self.standardize_data(X))
     eigen_values, eigen_vectors =  eig(Cov_mat)

     return (eigen_values, eigen_vectors)

  def fit(self,X):

      eigen_values, eigen_vectors = self.eigV(X)

      idx = np.array([np.abs(i) for i in eigen_values]).argsort()[::-1]

      eigen_values_sorted = eigen_values[idx]
      eigen_vectors_sorted = eigen_vectors.T[:,idx]

      explained_variance = [(i/sum(eigen_values))*100 for i in eigen_values_sorted]
      explained_variance = np.round(explained_variance, 2)
      cum_explained_variance = np.cumsum(explained_variance)

      explained_variance = np.round(explained_variance, 2)
      cum_explained_variance = np.cumsum(explained_variance)

      return eigen_vectors_sorted

  def transform(self):

      P = self.fit(X)[:self.n_component, :] # Projection matrix
      X_proj = X_std.dot(P.T)
      return X_proj
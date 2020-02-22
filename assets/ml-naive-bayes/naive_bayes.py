import numpy as np
from scipy.stats import norm, multivariate_normal
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


class NaiveBayes:

    def __init__(self):
        self.params = {}
    
    def fit(self, X, Y, epsilon=1e-2):
        """Fit NB classifier assuming a normal pdf for the likelihood

        Parameters
        ----------
        X : numpy.ndarray
            Training features. For MNIST, this is the pixel values
        Y : numpy.ndarray
            Target labels. For MNIST, this is the digits
        """
        for class_ in np.unique(Y):
            x_c = X[Y==class_]
            
            self.params[class_] = {
                'means': x_c.mean(axis=0),
                'std': x_c.var(axis=0) + epsilon,
                'prior': (Y==class_).mean(keepdims=True),
            }
                
    def predict(self, X):
        """Run inference on data

        Parameters
        ----------
        X : numpy.ndarray
            Data to predict on. dims 2: [number of cases, number of features]
        """
        N, _ = X.shape
        num_classes = len(self.params)
        log_posterior = np.zeros((N, num_classes))  # placeholder, we want to predict a class for each case
        
        # Calculate log{P(Y|X)} = sum_i log{P(x_i|Y)} + log{P(Y)}
        # We do this for all cases simultaneously
        for class_, pram in self.params.items():
            # log_liklehood = norm.logpdf(X, loc=pram['means'], scale=pram['std']).sum(axis=1)
            log_liklehood = multivariate_normal.logpdf(X, mean=pram['means'], cov=pram['std']) #.sum(axis=1)
            log_prior = np.log(pram['prior'])
            log_posterior[:, class_] = log_liklehood + log_prior

        return np.argmax(log_posterior, axis=1)
    
    def evaluate(self, X, Y):
        P = self.predict(X)
        return np.mean(Y == P)


print("Loading data... ")
X, Y = fetch_openml('mnist_784', version=1, return_X_y=True)
Y = Y.astype(int)
print("Done")

# Normalize with min-max scaling.
# The data does not need to be normalized; however, the smoothing parameter
# in training will have to change to compensate for this. If not normalizing,
# try epsilon = 255
X = (X - X.min()) / (X.max() - X.min())

xtrain, xtest, ytrain, ytest = train_test_split(X, Y)

nb = NaiveBayes()
nb.fit(xtrain, ytrain)
print("Accuracy on MNIST classification: {:.2f}%".format(100*nb.evaluate(xtest, ytest)))

import numpy as np

class Logistic(object):
    def __init__(self, d=784, reg_param=0):
        """"
        Inputs:
          - d: Number of features
          - regularization parameter reg_param
        Goal:
         - Initialize the weight vector self.w
         - Initialize the  regularization parameter self.reg
        """
        self.reg  = reg_param
        self.dim = [d+1, 1]
        self.w = np.zeros(self.dim)
    def gen_features(self, X):
        """
        Inputs:
         - X: A numpy array of shape (N,d) containing the data.
        Returns:
         - X_out an augmented training data to a feature vector e.g. [1, X].
        """
        N,d = X.shape
        X_out= np.zeros((N,d+1))
        # ================================================================ #
        # YOUR CODE HERE:
        # IMPLEMENT THE MATRIX X_out=[1, X]
        # ================================================================ #
        
        X_out = np.insert(X, 0, 1, axis=1)

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return X_out  
    def loss_and_grad(self, X, y):
        """
        Inputs:
        - X: N x d array of training data.
        - y: N x 1 labels 
        Returns:
        - loss: a real number represents the loss 
        - grad: a vector of the same dimensions as self.w containing the gradient of the loss with respect to self.w 
        """
        loss = 0.0
        grad = np.zeros_like(self.w) 
        N,d = X.shape 
        
        # ================================================================ #
        # YOUR CODE HERE:
        # Calculate the loss function of the logistic regression
        # save loss function in loss
        # Calculate the gradient and save it as grad
        # ================================================================ #
        X_with_bias = np.hstack([np.ones((N, 1)), X])
        
        a = np.dot(X_with_bias, self.w)
        h = 1 / (1 + np.exp(-a))
        
        loss = -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) / N
        grad = np.dot(X.T,(h - y)) / N

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return loss, grad
    
    def train_LR(self, X, y, eta=1e-3, batch_size=1, num_iters=1000) :
        """
        Inputs:
         - X         -- numpy array of shape (N,d), features
         - y         -- numpy array of shape (N,), labels
         - eta       -- float, learning rate
         - num_iters -- integer, maximum number of iterations
        Returns:
         - loss_history: vector containing the loss at each training iteration.
         - self.w: optimal weights 
        """
        loss_history = []
        N,d = X.shape
        self.w = np.zeros((d + 1, 1))
        X_with_bias = np.hstack([np.ones((N, 1)), X])
        for t in np.arange(num_iters):
            X_batch = None
            y_batch = None
            # ================================================================ #
            # YOUR CODE HERE:
            # Sample batch_size elements from the training data for use in gradient descent.  
            # After sampling, X_batch should have shape: (batch_size,1), y_batch should have shape: (batch_size,)
            # The indices should be randomly generated to reduce correlations in the dataset.  
            # Use np.random.choice.  It is better to user WITHOUT replacement.
            # evaluate loss and gradient for batch data
            # save loss as loss and gradient as grad
            # update the weights self.w
            # ================================================================ #
            indices = np.random.choice(N, batch_size, replace=False)
            X_batch = X_with_bias[indices]
            y_batch = y[indices]

            # obtain alpha
            a = np.dot(X_batch, self.w)

            # sigmoid function
            h = 1 / (1 + np.exp(-a))

            loss = -np.sum(y_batch * np.log(h) + (1 - y_batch) * np.log(1 - h)) / batch_size
            
            grad = np.dot(X_batch.T, h - y_batch) / batch_size
            self.w = self.w - eta*grad
            
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
            loss_history.append(loss)
        return loss_history, self.w
    
    def predict(self, X):
        """
        Inputs:
        - X: N x d array of training data.
        Returns:
        - y_pred: Predicted labelss for the data in X. y_pred is a 1-dimensional
          array of length N.
        """
        y_pred = np.zeros(X.shape[0]+1)
        # ================================================================ #
        # YOUR CODE HERE:
        # PREDICT THE LABELS OF X 
        # ================================================================ #
        X = self.gen_features(X)
        self.w = self.w.reshape(-1)
        
        a = np.dot(X,self.w)
        h = 1 / (1 + np.exp(-a)) 
        
        y_pred[h >= 0.5] = 1
        y_pred[h < 0.5] = -1
        

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        return y_pred
import numpy as np

class Regression(object):
    def __init__(self, m=1, reg_param=0):
        """"
        Inputs:
          - m Polynomial degree
          - regularization parameter reg_param
        Goal:
         - Initialize the weight vector self.w
         - Initialize the polynomial degree self.m
         - Initialize the  regularization parameter self.reg
        """
        self.m = m
        self.reg  = reg_param
        self.dim = [m+1 , 1]
        self.w = np.zeros(self.dim)
    def gen_poly_features(self, X):
        """
        Inputs:
         - X: A numpy array of shape (N,1) containing the data.
        Returns:
         - X_out an augmented training data to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].
        """
        N,d = X.shape
        m = self.m
        X_out= np.zeros((N,m+1))
        
        if m==1:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X]
            # ================================================================ #
            X_out = np.insert(X, 0, 1, axis=1)
            X_out[:, 1] = X.squeeze()
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        else:
            # ================================================================ #
            # YOUR CODE HERE:
            # IMPLEMENT THE MATRIX X_out=[1, X, x^2,....,X^m]
            # ================================================================ #
            X_out = np.insert(X, 0, 1, axis=1)
            for i in range(1, m + 1):
                X_out[:, i] = np.power(X, i).squeeze()
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
        return X_out  
    
    def loss_and_grad(self, X, y):
        """
        Inputs:
        - X: N x d array of training data.
        - y: N x 1 targets 
        Returns:
        - loss: a real number represents the loss 
        - grad: a vector of the same dimensions as self.w containing the gradient of the loss with respect to self.w 
        """
        loss = 0.0
        grad = np.zeros_like(self.w) 
        m = self.m
        N,d = X.shape 
        X_new = self.gen_poly_features(X)
        # ================================================================ #
        # YOUR CODE HERE:
        # Calculate the loss function of the linear regression
        # save loss function in loss
        # Calculate the gradient and save it as grad
        #
        # ================================================================ #
        h = np.dot(X_new,self.w)
        print(self.w)
        diff = h - y

        
        loss = np.sum(diff ** 2) + self.w * np.sum(self.w[1:] ** 2) / 2
    
        grad = np.sum(np.dotX_new.T, diff) + 2 * self.w * np.concatenate(([0], self.w[1:])) / N
        
        
        # END YOUR CODE HERE
        # ================================================================ #
        return loss, grad
    
    def train_LR(self, X, y, eta=1e-3, batch_size=30, num_iters=1000) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares batch gradient descent.

        Inputs:
         - X         -- numpy array of shape (N,1), features
         - y         -- numpy array of shape (N,), targets
         - eta       -- float, learning rate
         - num_iters -- integer, maximum number of iterations
         
        Returns:
         - loss_history: vector containing the loss at each training iteration.
         - self.w: optimal weights 
        """
        loss_history = []
        N,d = X.shape
        for t in np.arange(num_iters):
            X_batch = None
            y_batch = None
            # ================================================================ #
            # YOUR CODE HERE:
            # Sample batch_size elements from the training data for use in gradient descent.  
            # After sampling, X_batch should have shape: (batch_size,1), y_batch should have shape: (batch_size,)
            # The indices should be randomly generated to reduce correlations in the dataset.  
            # Use np.random.choice.  It is better to user WITHOUT replacement.
            # ================================================================ #
            indices = np.random.choice(N, batch_size, replace=False)
            X_batch = X[indices]
            y_batch = y[indices]
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
            loss = 0.0
            grad = np.zeros_like(self.w)
            # ================================================================ #
            # YOUR CODE HERE: 
            # evaluate loss and gradient for batch data
            # save loss as loss and gradient as grad
            # update the weights self.w
            # ================================================================ #
            X_new = self.gen_poly_features(X_batch)
            h = np.dot(X_new,self.w)
            error = h - y_batch
            loss = np.sum(error ** 2) / batch_size
            grad = 2 * X_new.T.dot(error) / batch_size

            self.w = self.w - (eta * grad)
            # ================================================================ #
            # END YOUR CODE HERE
            # ================================================================ #
            loss_history.append(loss)
        return loss_history, self.w
    def closed_form(self, X, y):
        """
        Inputs:
        - X: N x 1 array of training data.
        - y: N x 1 array of targets
        Returns:
        - self.w: optimal weights 
        """
        m = self.m
        N,d = X.shape
        # ================================================================ #
        # YOUR CODE HERE:
        # obtain the optimal weights from the closed form solution 
        # ================================================================ #
        X_new = self.gen_poly_features(X)
        self.w = np.linalg.inv(X_new.T @ X_new) @ X_new.T @ y
        
        y_pred = X_new @ self.w
        error = y - y_pred
        loss = np.sum(error ** 2) / N
        # ================================================================ #
        # END YOUR CODE HERE

        return loss, self.w
    
    
    def predict(self, X):
        """
        Inputs:
        - X: N x 1 array of training data.
        Returns:
        - y_pred: Predicted targets for the data in X. y_pred is a 1-dimensional
          array of length N.
        """
        y_pred = np.zeros(X.shape[0])
        m = self.m
        
        X_new = self.gen_poly_features(X)
        y_pred = np.dot(X_new, self.w)
        
        return y_pred
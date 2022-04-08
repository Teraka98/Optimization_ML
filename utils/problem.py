import numpy as np


class Problem():

    def __init__(self, X, y, lbda, loss="l2") -> None:
        self.lbda = lbda
        self.X = X
        self.y = y
        self.n, self.d = X.shape
        self.loss = loss

    
    
    #### For the second part we are going to set fi, grad_fi, hessian_fi
    def f_i(self, i, w):
        yXwi = self.y[i] * np.dot(self.X[i], w)
        if self.loss=='l2':
            return np.log(1. + np.exp(- yXwi)) + self.lbda * np.linalg.norm(w) ** 2 / 2
        elif self.loss ==  "logit":
            yXwi = self.y[i] * np.dot(self.X[i], w)
            return np.log(1. + np.exp(- yXwi)) + self.lbda * np.linalg.norm(w) ** 2 / 2.

    
        
    def grad_i(self, i, w):
        x_i = self.X[i]
        if self.loss == "l2":
            grad = - x_i * self.y[i] / (1. + np.exp(self.y[i]* x_i.dot(w)))
            grad += self.lbda * w
            return grad
        elif self.loss == "logit":
            x_i = self.X[i]
            grad = - x_i * self.y[i] / (1. + np.exp(self.y[i]* x_i.dot(w)))
            grad += self.lbda * w
            return grad
        
    
    
    def hessian_i(self, w, i):
        yXwi = self.y[i] * np.dot(self.X[i], w)

        hessian = np.exp(yXwi) / (1.0 + np.exp(yXwi))
        hessian = hessian * np.dot(self.X[i], self.X[i].T)
        hessian = hessian + self.lbda * np.eye(w.shape[0])
        return hessian
    
        

    
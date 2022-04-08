from tkinter import E
import numpy as np
from utils.search import LineSearch


class Newton:
    
    def __init__(self, w0, epsilon) -> None:
        self.w0 = w0
        self.epsilon = epsilon

    def newtown_methods(self, f, grad, hessian, eps=10e-6):
        """
        
        """
        
        w = self.w0.copy()
        
        k = 1

        
        while np.linalg.norm(f(w)) > eps:
            
            w = self.w0 - np.linalg.pinv(hessian(w)) @ grad(w)
            self.w0 = w
            k = k+1
        
        print("It converges after %s iterations for Classic methods" %k)

        return w

        
    def newton_global(self, f, hessian, grad, c, theta):
        """
        
        """

        w = self.w0.copy()
        k = 1
        search = LineSearch()

        while np.linalg.norm(f(w)) > self.epsilon:


            dk = search.get_direction(hessian, w, grad, False)
            alpha = search.armijo_rule(f, c, grad, w, theta, dk)
            w = self.w0 + alpha*dk
            self.w0 = w
            k += 1
            
        
        print("It converges after %s iterations for Global Newton" %k)


        return w

    def bgfs(self, f, grad, hessian, theta, c):
        """
        
        """

        wk = self.w0.copy()
        h0 = np.eye(hessian(wk).shape[0])
        id = np.eye(wk.shape[0])
        k = 1

        search = LineSearch()



        while np.linalg.norm(grad(wk)) > self.epsilon :

            k += 1

            ## -------- Compute sk and vk ----------- ##

            #1. Compute wk+1
            dk = search.get_direction(hessian, wk, grad, False)

            alpha_ = search.armijo_rule(f, c, grad, wk, theta, dk)


            wk = self.w0 - alpha_ * (h0 @ grad(self.w0))


            sk = wk - self.w0

            vk = np.array(grad(wk)) - np.array(grad(self.w0))
            
            # ## --------- Update hk ----------------##
            cond = sk.T @ vk
            if cond > 0:
                val = (vk @ sk.T) / (sk.T @ vk)
                term = id - val
                hk = (term.T @ h0 @ term) + (sk @ sk.T/cond)

                h0 = hk
            else:
                h0 = h0

            
            self.w0 = wk
            
        
        print("It converges after %s iterations for BGFS" %k)

        return wk
    
    def l_bgfs(self, f, grad, hessian, theta, c, m):
        """
        
        """
        wk = self.w0.copy()
        h0 = np.eye(hessian(wk).shape[0])
        id = np.eye(wk.shape[0])
        k = 0

        sk_list = [np.zeros(wk.shape) for i in range(m)]
        vk_list = [np.zeros(wk.shape) for i in range(m)]

        end = 0
        search = LineSearch()
        while np.linalg.norm(wk) > self.epsilon :

            ## -------- Compute sk and vk ----------- ##

            #1. Compute wk+1
            dk = search.get_direction(hessian, wk, grad, False)

            alpha_ = search.armijo_rule(f, c, grad, wk, theta, dk)


            wk = self.w0 - alpha_ * (h0 @ grad(self.w0))



            sk = wk - self.w0


            sk_list[end] = sk

            vk = np.array(grad(wk)) - np.array(grad(self.w0))
            
            vk_list[end] = vk

            # _iter_ = max(0, m-1)
            # print("LEN",len(vk_list))

            bound = (m <= k and [m] or [k])[0]
            k = k + 1
			# k = k + 1
			# # get the next one in the last m iteration data list
            end = (end + 1) % m
            j = end
           

            for i in range(0, bound):
                j = (j + m -1) % m
                
                print(k-j)
                # ## --------- Update hk ----------------##
                cond = sk_list[k-(j)].T @ vk_list[k-(j)]
                if cond > 0:
                    val = (vk_list[k-(j-1)] @ sk[k-(j-1)].T) / (sk[k-(j-1)].T @ vk[k-(j-1)])
                    term = id - val
                    hk = (term.T @ h0 @ term) + (sk_list[k-(j-1)] @ sk[k-(j-1)].T/cond)
                    h0 = hk
                else:
                    h0 = h0
                
                
            self.w0 = wk

            k += 1 

        return wk


    
    



    

    
    
    
    


        





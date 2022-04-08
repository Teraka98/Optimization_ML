from cmath import inf
import numpy as np
import math as mt

class LineSearch(object):

    
    def __init__(self) -> None:
        pass

    def get_direction(self, hessian, wk, grad, is_sampling):

        if is_sampling:

            num = np.linalg.pinv(hessian)


            direction_k = -num @ grad


            return direction_k

        else:

            eigen_val = np.linalg.eigvalsh(hessian(wk))


            lamda_k = 2 * max(-(np.amin(eigen_val)), 10e-10)


            num = np.linalg.pinv(hessian(wk) + lamda_k * np.eye(hessian(wk).shape[0]))


            direction_k = -num @ grad(wk)


            return direction_k
    

    def armijo_rule(self, f, c, grad, wk, theta, d):

        test = 1
        j = 1
        alpha_ = theta**j


        while test:
            x_new = wk + alpha_ * d
    
            if f(x_new)< f(wk) + c*alpha_*np.dot(d.T, grad(wk)):
                test = 0
            else:
                j += 1
                alpha_ = theta **j

        return mt.ceil(alpha_)

    def armijo_samling(self, pb, wk, grad, c, theta, nb, dk):

        j = 1
    
        alpha_ = theta**j

        d = wk.shape[0]

        n_samples = pb.n

        ik = np.random.choice(n_samples, nb, replace=False)

        test = 1

        f_new = np.zeros(d)
        cond = np.zeros(d)
        
        while test:
            x_new = wk + alpha_ * dk
            for k in range(nb):
                fi_new = pb.f_i(ik[k], x_new)
                
                cond_i = pb.f_i(ik[k], wk) + c*alpha_*np.dot(dk.T, grad)
                
                f_new = f_new + fi_new
                cond = cond + cond_i
            
            
            f_new = (1/nb)*f_new
            f_new_mean = np.mean(f_new)
            cond = (1/nb)*cond
            cond_mean = np.mean(cond)


            if f_new_mean == inf:
                return 1
            
            if f_new_mean < cond_mean:
                test = 0
                break
            else:
                j += 1
                alpha_ = theta **j

        
        
        return alpha_
                    

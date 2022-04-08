from xml.dom.domreg import well_known_implementations
import numpy as np

from utils.search import LineSearch


class Method:

    def __init__(self, X, w0, nb, lbda, epsilon) -> None:
        self.w0 = w0
        self.nb = nb
        self.epsilon = epsilon
        self.X = X
        self.lbda = lbda



    def get_stocastic(self, w, pb, ik):

        d = w.shape[0]


        # Compute the stochastic grad and hessian
        stochastic_grad = np.zeros(d)
        stochastic_hess = np.zeros(d)

        for j in range(self.nb):

            #Compute gradients
            gi = pb.grad_i(ik[j], w)
            stochastic_grad = stochastic_grad + gi
            
            # #Compute hessian
            hi = pb.hessian_i(w, ik[j])
            stochastic_hess = stochastic_hess + hi


        stochastic_grad = (1/self.nb)*stochastic_grad
        stochastic_hess = (1/self.nb)*stochastic_hess

        return stochastic_grad, stochastic_hess

    

    def subsampling_newton(self, pb, c, theta, alpha_user, step_choice, n_iter=100):
        """
        step_choice :
            - 0 : Apply Armijo rule
            - 1 : Compute alpha_ with L_sk
            - 2 : Comput alpha with L
        """

        w0 = self.w0.copy()

        w = self.w0.copy()

        search = LineSearch()

        k = 0

        n_samples = pb.n

        nw = list()


        while k < n_iter and  np.linalg.norm(w) < self.epsilon:

            ik = np.random.choice(n_samples, self.nb, replace=False)

            grad_sh, hess_sh = self.get_stocastic(w, pb, ik)

            direction_k = search.get_direction(hessian=hess_sh, wk=w, grad=grad_sh, is_sampling=True)
            
            if step_choice == 0:
                #Armijo Rule
                alpha_ = search.armijo_samling(pb, w, grad_sh, c, theta, self.nb, direction_k)
            elif step_choice == 1:
                #Compute Lsk
                sum_norm = 0
                for s in range(self.nb):
                    sum_norm += np.linalg.norm(self.X[ik] @ self.X[ik].T)
                #Compute alpha_
                L_sk = ((4*sum_norm)/self.nb) + self.lbda
                alpha_ = alpha_user/L_sk
            
            elif step_choice == 2:
                #Compute Ls
                Ls = ((4*np.linalg.norm(self.X @ self.X.T))/n_samples) + self.lbda
                alpha_ = alpha_user/Ls

            else:
                alpha_ = alpha_user

            w = w0 + alpha_ * direction_k

            nw.append(np.linalg.norm(w0))


            w0 = w

            k += 1




        return w, nw, n_iter
    

    def bgfs_subsampling(self, problem, c, theta, alpha_user, step_choice, n_iter=100):
        """
        
        """
        
        
        wk = self.w0.copy()
        h0 = np.eye(wk.shape[0])
        id = np.eye(wk.shape[0])
        k = 1
        n_samples = problem.n


        search = LineSearch()

        ik = np.random.choice(n_samples, self.nb, replace=False)


        grad, hess = self.get_stocastic(wk, problem, ik)

        nw = list()
        

        while k< n_iter and np.linalg.norm(grad) < self.epsilon :

            ik = np.random.choice(n_samples, self.nb, replace=False)

            grad, hess = self.get_stocastic(wk, problem, ik)

            k += 1

            ## -------- Compute sk and vk ----------- ##

            #1. Compute wk+1
            dk = search.get_direction(hess, wk, grad, True)

            if step_choice == 0:
                #Armijo Rule
                alpha_ = search.armijo_samling(problem, wk, grad, c, theta, self.nb, dk)
            elif step_choice == 1:
                #Compute Lsk
                sum_norm = 0
                for s in range(self.nb):
                    sum_norm += np.linalg.norm(self.X[ik] @ self.X[ik].T)
                #Compute alpha_
                L_sk = ((4*sum_norm)/self.nb) + self.lbda
                alpha_ = alpha_user/L_sk
            
            elif step_choice == 2:
                #Compute Ls
                Ls = ((4*np.linalg.norm(self.X @ self.X.T))/n_samples) + self.lbda
                alpha_ = alpha_user/Ls

            else:
                alpha_ = alpha_user

            alpha_ = search.armijo_samling(problem, wk, grad, c, theta, self.nb, dk)


            wk = self.w0 - alpha_ * (h0 @ grad)


            sk = wk - self.w0

            vk = np.array(grad) - np.array(grad)
            
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

            nw.append(np.linalg.norm(self.w0))

            
        

        return wk, nw, n_iter





        
        
    

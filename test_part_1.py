import numpy as np
import time


from src.part_1 import Newton


def func_test(wk):
  return 2*(wk[0] + wk[1] + wk[2] - 3)**2 + (wk[0] - wk[1])**2 + (wk[1] - wk[2])**2

def grad_test(wk):
  return [2*(3*wk[0] + wk[1] + 2*wk[2] - 6), 2*(wk[0] + 4*wk[1] + wk[2] - 6), 2*(2*wk[0] + wk[1] + 3*wk[2] - 6)]

def hessian_test(wk):
  return np.array([

    [6, 2, 4],
    [2, 8, 2],
    [4, 2, 6]
  ])

############# Problem 2 #########
  
def func_(wk):
  return  (100*(wk[1]-wk[0]**2)**2 + (1-wk[0])**2 )

def jac(wk):
  return [400*wk[0]**3 + 2*wk[0] - 400*wk[0]*wk[1] -2, -200*wk[0]**2 + 200*wk[1]]

def hessian(wk):
  return np.array([
    [-400*(wk[1]-wk[0]**2) + 800*wk[0]**2 + 2, -400*wk[1]], 
    [-400*wk[0], 200]
    ], dtype=float)



    
  
  
  
###### Test : Pass this value as arguement for the first question ######  
w0_first = np.array([
          [1], [1], [1]
                    ])

w01 = np.array([
  [-1.2], [1]
])

#Hard to tune epsilon for this value -> We cannot say if it converging or not
w02 = np.array([
  [0], [1/200 + 10e-12]
])

w03 = np.array([
  [0], [5e-3]
])

## Change w0 starting point here
method = Newton(w03, 10e-6)


# Question 1.1 -- Please comment for the next question as we are testing vector with size (2, 2)
# w_ros = method.newtown_methods(func_test, grad_test, hessian_test)
# print(w_ros)

# Question 1.2
start_time = time.time()
w_ros = method.newtown_methods(func_, jac, hessian)
print("values for classic newton:\n", w_ros)
print("--- %s seconds ---" % (time.time() - start_time))



start_time = time.time()
w = method.newton_global(func_, hessian, jac, 0.00005, 0.00009)
print("value for global newton:\n", w)
print("--- %s seconds ---" % (time.time() - start_time))


sk = method.bgfs(func_, jac, hessian, 0.5, 0.0001)
print("value for bgfs:\n", sk)

# Question 1.1 -- Uncomment for the next question as we are testing in (2, 2)
# sk = method.bgfs(func_test, grad_test, hessian_test, 0.5, 0.0001)
# print("value for bgfs:\n", sk)

# sk = method.l_bgfs(func_, jac, hessian, 0.5, 0.0001, 5)
# print(sk)




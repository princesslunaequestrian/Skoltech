#!/usr/bin/env python
# coding: utf-8

# # Problem Set 2

# ## Problem 1

# Consider linear system
# 
# $$
# \begin{align*}
# 2x_1 + x_2 = 1\\
# x_1 + 2x_2 + x_3 = 2\\
# x_2 + 2x_3 = 3
# \end{align*}
# $$
# 
# (a) Find the $LU$ factorization of the coefficient matrix $A$.   Show that $U = DL^T$ with $D$ diagonal and thus $A=LDL^T$. Find the exact solution using the $LU$ factorization.  
# 
# (b) Solve the system using Jacobi and Gauss-Seidel iterations. How many iterations are needed to reduce the relative error of the solution to $10^{-8}$?  
# 
# (c) Plot in semilog scales the relative errors by both methods as a function of the number of iterations.
# 
# (d) Explain the convergence rate. Which of the methods is better and why?
# 
# 

# ### Solution
# 
# Let us perform the $LU$ factorization by hand:

# 
# $$
# \begin{align*}
# \begin{bmatrix}
# 2 & 1 & 0\\
# 1 & 2 & 1\\
# 0 & 1 & 2
# \end{bmatrix}
# \Rightarrow (R_2 - \frac{1}{2}R_1)
# \begin{bmatrix}
# 2 & 1 & 0\\
# 0 & \frac{3}{2} & 1\\
# 0 & 1 & 2
# \end{bmatrix}
# \Rightarrow (R_3 - \frac{2}{3}R_2)
# \begin{bmatrix}
# 2 & 1 & 0\\
# 0 & \frac{3}{2} & 1\\
# 0 & 0 & \frac{4}{3}
# \end{bmatrix}
# = U;
# \hspace{10mm}
# L =
# \begin{bmatrix}
# 1 & 0 & 0\\
# \frac{1}{2} & 1 & 0\\
# 0 & \frac{2}{3} & 1
# \end{bmatrix}
# \end{align*}
# $$

# And check it via *numpy*:

# In[1]:


import numpy as np
from fractions import Fraction
from scipy import linalg as lin
import sympy as sp
from sympy import abc

A = np.matrix([
    [2, 1, 0],
    [1, 2, 1],
    [0, 1, 2]
])

def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)

E, L, U = lin.lu(A, permute_l = False)
L = np.matrix(L)
U = np.matrix(U)
print("L:")
print(L)
print("\nU:")
print(U)


# Let us show, that $U = DL^T$:
# 
# $$
# L^T = 
# \begin{bmatrix}
# 1 & \frac{1}{2} & 0\\
# 0 & 1 & \frac{2}{3}\\
# 0 & 0 & 1
# \end{bmatrix}
# ;\hspace{10mm}
# D = 
# \begin{bmatrix}
# 2 & 0 & 0\\
# 0 & \frac{3}{2} & 0\\
# 0 & 0 & 1
# \end{bmatrix}
# ;\hspace{10mm}
# U = DL^T
# $$
# 
# That means, $A = LU = LDL^T$.

# Now, let us solve the system using $LU$ factorization:
# 
# $$
# Ax = b \hspace{3mm} \Rightarrow \hspace{3mm} LUx = b;
# \\
# \begin{align*}
# \begin{cases}
# Ux = y;\\
# Ly = b
# \end{cases}
# \end{align*}
# $$

# $$
# \begin{bmatrix}
#   1 & 0 & 0\\
#   \frac{1}{2} & 1 & 0\\
#   0 & \frac{2}{3} & 1\\
# \end{bmatrix}
# \begin{bmatrix}
#   y_1\\
#   y_2\\
#   y_3
# \end{bmatrix}
# =
# \begin{bmatrix}
# 1\\
# 2\\
# 3
# \end{bmatrix}
# \Rightarrow
# \begin{cases}
# y_1 = 1\\
# y_2 = \frac{3}{2}\\
# y_3 = 2
# \end{cases}
# $$
# 
# $$
# \begin{bmatrix}
# 2 & 1 & 0\\
# 0 & \frac{3}{2} & 1\\
# 0 & 0 & \frac{4}{3}
# \end{bmatrix}
# \begin{bmatrix}
# x_1\\
# x_2\\
# x_3
# \end{bmatrix}
# =
# \begin{bmatrix}
# 1\\
# \frac{3}{2}\\
# 2
# \end{bmatrix}
# \Rightarrow
# \begin{cases}
# x_1 = \frac{1}{2}\\
# x_2 = 0\\
# x_3 = \frac{3}{2}
# \end{cases}
# $$
# 
# $$
# x = 
# \begin{bmatrix}
# 1/2\\
# 0\\
# 3/2
# \end{bmatrix}
# $$

# Next we solve the system using Jacobi and Gauss-Seidel methods:

# In[2]:


import matplotlib.pyplot as plt
import copy


# In[3]:


D = np.matrix(np.diag(np.diag(A)))
U = np.matrix(np.triu(A-D))
L = np.matrix(np.tril(A-D))

x_exact = np.matrix([1/2, 0, 3/2]).T
tol = 1e-8
err = 1
x_init = np.matrix([1, 1, 1]).T #initial x
b = np.matrix([1, 2, 3]).T

def Jacobi(L, D, U, x_init, b, err, tol):

    x = copy.deepcopy(x_init)

    max_iters = 500

    err_iter = np.array([])
    err_exact_iter = np.array([])
    

    iters = 0
    while ((err > tol) and (iters <= max_iters)) :
        iters = iters + 1
        bb = b - (U + L)*x
        x_new = np.linalg.solve(D, bb)
        err = np.linalg.norm(x_new - x)/np.linalg.norm(x)
        err_exact = np.linalg.norm(x_new - x_exact)

        err_iter = np.append(err_iter, err)
        err_exact_iter = np.append(err_exact_iter, err_exact)

        x = x_new

    return err_iter, err_exact_iter, x_new, iters

def Gauss_Seidel(L, D, U, x_init, b, err, tol):

    x = copy.deepcopy(x_init)

    max_iters = 250

    err_iter = np.array([])
    err_exact_iter = np.array([])

    iters = 0
    while ((err > tol) and (iters <= max_iters)):
        iters = iters + 1

        bb = b - U*x
        x_new = np.linalg.solve(D+L, bb)
        err = np.linalg.norm(x_new - x)/np.linalg.norm(x)
        err_exact = np.linalg.norm(x_new - x_exact)

        err_iter = np.append(err_iter, err)
        err_exact_iter = np.append(err_exact_iter, err_exact)

        x = x_new

    return err_iter, err_exact_iter, x_new, iters


# In[4]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format='svg'")

fig = plt.figure()
ax = plt.gca()

errs_j, ex_errs_j, x_j, i_j = Jacobi(L, D, U, x_init, b, err, tol)
plt.plot(errs_j)

errs_g, ex_errs_g, x_g, i_g = Gauss_Seidel(L, D, U, x_init, b, err, tol)
plt.plot(errs_g)

ax.set_yscale('log')


ax.legend(('Jacobi', 'Gauss_Seidel'))
ax.set_xlabel('Iterations')
ax.set_ylabel('Relative error')
ax.set_yticks([1e-7, 1e-5, 1e-3, 1e-1])
ax.tick_params(axis='y', which='minor')
ax.grid(which='both')


# As we see, it took $26$ iterations for Gauss-Seidel method to reach the target tolerance of $10^{-8}$, while Jacobi method required $54$ iterations.
# 
# This can be explained by the following: in Gauss-Seidel, as soon as we acquire a new iteration of a vector $x$ component $x_i^{(k+1)}$, we instantly utilize this updated value in the computation of the following components: $x_i^{(k+1)} = f(x_1^{(k+1)}, ..., x_{i-1}^{(k+1)}, x_{i+1}^{(k)}, ..., x_n^{(k)})$. In Jacobi, we calculate the new vector $x^{k+1}$ relying solely on the result of the previous iteration: $x_i^{k+1} = f(x_j^{(k)}),\hspace{3mm}j \neq i$.

# ## Problem 2

# Factor these two matrices $A$ into $S\Lambda S^{-1}$:
# 
# $$
# A1 = 
# \begin{bmatrix}
# 1 & 2\\
# 0 & 3
# \end{bmatrix}
# ,\hspace{3mm}
# A2 =
# \begin{bmatrix}
# 1 & 2\\
# 0 & 3
# \end{bmatrix}
# $$
# 
# Using that factorization, find for both: (a) $A^3$; (b) $A^{-1}$.

# ### Solution
# 
# Firstly, we find the eigenvalues and eigenvectors:
# 
# $$
# \begin{bmatrix}
# 1-\lambda & 2\\
# 0 & 3-\lambda
# \end{bmatrix}
#  = 0
# $$
# 
# By performing simple calculations by hand, we obtain:
# 
# $$
# h_1 = 
# \begin{bmatrix}
# 1\\
# 0
# \end{bmatrix}
# ,\hspace{3mm}
# \lambda_1 = 1
# \\
# h_2 =
# \frac{\sqrt{2}}{2}
# \begin{bmatrix}
# 1\\
# 1
# \end{bmatrix}
# ,\hspace{3mm}
# \lambda_2 = 3
# $$
# 
# ```{Note}
# We normalized the eigenvectors
# ```
# 
# Now let's perform a check with *numpy*:

# In[5]:


A1 = np.matrix([
    [1, 2],
    [0, 3]
])

val1, vec1 = np.linalg.eig(A1)
print("E_values A1:")
print(val1)
print("E_vectors A1:")
print(vec1)


# As we have our vectors and values, we can construct $\Lambda$ and $S$, $S^{-1}$ matrices:

# In[6]:


Lambda = np.matrix(np.diag(val1))
S = vec1
Si = np.linalg.inv(S)

print("S:")
print(S)
print("Lambda:")
print(Lambda)
print("S_inverse:")
print(Si)


# $$
# 
# S = 
# \begin{bmatrix}
# 1 & \sqrt{2}/2\\
# 0 & \sqrt{2}/{2}
# \end{bmatrix}
# ;\hspace{3mm}
# \Lambda = 
# \begin{bmatrix}
# 1 & 0\\
# 0 & 3
# \end{bmatrix}
# ;\hspace{3mm}
# S^{-1} = 
# \begin{bmatrix}
# 1 & -1\\
# 0 & 2/\sqrt{2}
# \end{bmatrix}
# $$

# From now, we can fing the $A^3$ powered matrix by simply multiplying the decomposition:
# 
# $$
# A^3 = S\Lambda S^{-1} S \Lambda S^{-1} S \Lambda S^{-1} =
# \\
# \hspace{1mm}
# \\
# = S \Lambda^3 S^{-1}.
# $$
# 
# And for $\Lambda$ it is easy to power because it is a diagonal matrix.
# 
# $$
# \Lambda ^3 =
# \begin{bmatrix}
# 1 & 0\\
# 0 & 9
# \end{bmatrix}
# ,
# \\
# \hspace{1mm}
# \\
# S\Lambda ^3 S^{-1} = 
# \begin{bmatrix}
# 1 & 8\\
# 0 & 9
# \end{bmatrix}
# $$
# 
# For inverse matrix:
# 
# $$
# S\Lambda S^{-1} A_1^{-1} = E, \Rightarrow A_1^{-1} = S^{-1} \Lambda ^{-1} S
# $$
# 
# We already have $S$ and $S^{-1}$, and for diagonal $\Lambda$ the inverse matrix contains the inverse diagonal elements of $\Lambda$:
# 
# $$
# \Lambda ^{-1}= 
# \begin{bmatrix}
# 1 & 0\\
# 0 & 1/3
# \end{bmatrix}
# $$
# 
# So we easily find $A_1^{-1}$:
# 
# $$
# A_1^{-1} = 
# \begin{bmatrix}
# 1 & \sqrt{2}{3}\\
# 0 & 1/3
# \end{bmatrix}
# $$

# Now let's look at the second matrix $A_2$. Instantly we notice it is a rank-1 matrix, thus, $A_2^{-1}$ matrix doesn't exist.
# 
# $$
# h_1 = 
# \begin{bmatrix}
# 1\\
# -1
# \end{bmatrix}
# ,\hspace{3mm}
# \lambda_1 = 0
# \\
# \hspace{1mm}
# \\
# h_2 =
# \begin{bmatrix}
# 1\\
# 3
# \end{bmatrix}
# ,\hspace{3mm}
# \lambda_2 = 4
# $$
# 
# It's eigenvectors are non-collinear and form a basis in 2-dimensional space. Thus we can perform the factorization.
# 
# $$
# S = 
# \begin{bmatrix}
# 1 & 1\\
# -1 & 3
# \end{bmatrix}
# ;\hspace{3mm}
# \Lambda = 
# \begin{bmatrix}
# 0 & 0\\
# 0 & 4
# \end{bmatrix}
# ;\hspace{3mm}
# S^{-1} =  
# \begin{bmatrix}
# 3/4 & -1/4\\
# 1/4 & 1/4
# \end{bmatrix}
# $$

# For $A_2^3$:
# 
# $$
# A_2^3 = 
# \begin{bmatrix}
# 64 & 64\\
# 192 & 192
# \end{bmatrix}
# .
# $$
# 

# ## Problem 3
# 
# Given a system $Ax = b$ with
# 
# $$
# A = 
# \begin{bmatrix}
#   1 & -1 & -3\\
#   2 & 3 & 4\\
#   -2 & 1 & 4\\
# \end{bmatrix}
# ,\hspace{3mm}
# b = 
# \begin{bmatrix}
#   3\\
#   a\\
#   -1\\
# \end{bmatrix}
# ,
# $$
# 
# for which $a$ there is a solution? Find the general solution of the system for that $a$.
# 
# 
# 
# 
# 

# ### Solution
# 
# Let's check the matrix' rank:

# In[7]:


a = abc.symbols('a')

A = sp.Matrix([
    [1, -1, -3],
    [2, 3, 4],
    [-2, 1, 4]
])

b = sp.Matrix([
    3, a, -1]
)

print('Rank: {}'.format(A.rank()))
A.rref()[0]


# This matrix is a rank-2 matrix. Let's find it's left nullspace, write down the solvability condition and find the appropriate $a$.
# Starting with the left null-space:

# In[8]:


y = A.T.nullspace()[0]
y


# We want to fulfill the following condition:
# 
# $$
# y^Tb = 0
# $$

# In[9]:


ans = sp.solve(y.T*b, a)
ans


# As we see, the system is solvable with $a = -19$. Let us perform some check:

# In[10]:


bs = b.subs(a, -19)
y.T*bs


# Now as we have our vector $b$ with wich the system is solvable, we may find the general solution for the system:

# In[11]:


a, b, c = abc.symbols('a b c')
system = A, bs
sol = sp.linsolve((A, bs), a, b, c); sol


# With $c \in \mathbb{R}$ we get our solution:
# 
# $$
# x =
# \begin{bmatrix}
# -2\\
# -5\\
# 0
# \end{bmatrix}
# +
# \begin{bmatrix}
# 1\\
# -2\\
# 1
# \end{bmatrix}
# \cdot
# c.
# $$

# ## Problem 4
# 

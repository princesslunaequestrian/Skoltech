#!/usr/bin/env python
# coding: utf-8

# ## Mathematics Methods in Engineering and Applied Science
# 
# # Problem Set 1
# 
# ##### By Buchnev Arseniy

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sympy import pretty, abc


# ### Problem 1
# ##### Some basic problems on matrix/vector multiplication.
# 
# (a) Calculate by hand the following matrix/vector products:
# $$
# (i) \hspace{10mm}
# \begin{bmatrix}
# 2 & 1\\
# \end{bmatrix}
# \begin{bmatrix}
# 1 & −2 & 1\\
# 2 & 3 & 2
# \end{bmatrix};
# \\
# \\
# (ii) \hspace{10mm}
# \begin{bmatrix}
# 2 & -1 & 1\\
# 3 & 0 & 4
# \end{bmatrix}
# \begin{bmatrix}
# 2 & 1\\
# 1 & 2\\
# -1 & 0
# \end{bmatrix}
# $$
# as a combination of columns of the left matrix aswell as a combination of rows of the right matrix.
# 
# (b) Write down a permutation matrix $P_4$ that exchanges row 1 with row 3 and row 2with row 4. What is the connection of this matrix with the permutation matricesthat exchange only row 1 and row 3, and only row 2 and row 4?

# #### Solution
# <img src='./problem1.jpg' width=800>

# ### Problem 2
# 
# Given a $3\times3$ matrix $A= \begin{bmatrix}a1&a2&a3\end{bmatrix}$ with columns $a_i$, find a matrix $B$ that when multiplied with $A$, either from left or right, performs the following operations with $A$:
# 
# (a) exchanges row 1 and row 2;  
# (b) exchanges columns 1 and 2;  
# (c) doubles the first row;  
# (d) subtracts twice row 1 from row 2.  
# Also find the inverse of this matrix. What doesthe inverse of this $B$ do?

# #### Solution
# <img src='./problem2 (1).jpg' width=800>
# <img src='./problem2 (2).jpg' width=800>

# ### Problem 3
# 
# 
# For matrix 
# A =
# $\begin{bmatrix}
# 1 & 2 & 3\\
# 3 & 4 & 5\\
# 5 & 6 & 7
# \end{bmatrix}$
# , determine the following:
# 
# (a) rank;  
# (b) eigenvalues and eigenvectors;  
# (c) nullspace and left nullspace;  
# (d) column space and row space;  
# (e) write $A$ as a sum of rank-1 matrices in at least two different ways.

# #### Solution
# 
# <img src='./problem3 (1).jpg' width=800>
# <img src='./problem3 (2).jpg' width=800>

# ***d) Column space and row space:***
# 
# * Column space: $span(
# \begin{bmatrix}
# 1\\
# 0\\
# -1
# \end{bmatrix},
# \begin{bmatrix}
# 0\\
# 1\\
# 2
# \end{bmatrix})
# $; equals to the row space.
# 
# ***e) write $A$  as a sum of rank-1 matrices in at least two different ways:***
# 
# Obviously, we can present the matrix $A$ as a sum of single-row or single-column rank-1 matricies (with other rows or columns filled with zeros). Also, we have 3 non-colinear eigenvectors, that can be our matrix basis in the canonical decomposition (diagonalization):
# 
# $$
# A = UDU^{-1} = \sum\limits_{i=1}\limits^{3}{d_i\lambda_i\lambda_i^T}
# $$

# ### Problem 4
# 
# The columns of matrix $C = 
# \begin{bmatrix}
# 2 & 2 & 1 & 1 & 2 & 2 & 1 & 1\\
# 1 & 2 & 2 & 1 & 1 & 2 & 2 & 1\\
# 1 & 1 & 1 & 1 & 2 & 2 & 2 & 2
# \end{bmatrix}
# $
# represent vertices of a cube. Describe transformations of the cube that result from the action on $C$ of the following three matrices:
# 
# $A_1 = 
# \begin{bmatrix}
# 1 & 2 & 2\\
# 0 & 2 & 2\\
# 0 & 0 & 3
# \end{bmatrix}, \hspace{3mm}
# A_2 = 
# \begin{bmatrix}
# 1 & 2 & 2\\
# 0 & 2 & 2\\
# 0 & 0 & 0
# \end{bmatrix}, \hspace{3mm}
# A3 = 
# \begin{bmatrix}
# 0 & 2 & 2\\
# 0 & 2 & 2\\
# 0 & 0 & 0
# \end{bmatrix}.
# $
# 
# Relate the results to the ranks of $A_k$ and to the dimensions and bases of the four fundamental subspaces of $A_k$. Is there a $3\times3$ matrix $A$ that can transform a cube into a tetrahedron? Explain.

# #### Solution
# 
# Firstly, let us see what every transformation does with the cube.

# In[15]:


# Creating matricies

A1 = np.matrix([
    [1, 2, 2],
    [0, 2, 2],
    [0, 0, 3]
])

A2 = np.matrix([
    [1, 2, 2],
    [0, 2, 2],
    [0, 0, 0]
])

A3 = np.matrix([
    [0, 2, 2],
    [0, 2, 2],
    [0, 0, 0]
])

C = np.matrix([
    [2, 2, 1, 1, 2, 2, 1, 1],
    [1, 2, 2, 1, 1, 2, 2, 1],
    [1, 1, 1, 1, 2, 2, 2, 2]
])

#Applying A matricies to the cube

B1 = A1*C
B2 = A2*C
B3 = A3*C

#Plotting

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs = C[0], ys = C[1], zs = C[2])
ax.scatter(xs = B1[0], ys = B1[1], zs = B1[2], color='red', depthshade = True)
ax.scatter(xs = B2[0], ys = B2[1], zs = B2[2], color='green', depthshade = True)
ax.scatter(xs = B3[0], ys = B3[1], zs = B3[2], color='yellow', depthshade = True)
ax.set_xticks(np.arange(0, 11, 2))
ax.set_yticks(np.arange(0, 11, 2))
ax.set_zticks(np.arange(0, 7, 1))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.title('Cube Transformations')

plt.legend(('Initial Cube', 'A1', 'A2', 'A3'))


# The points on this 3D plot represent the verticies of the initial cube (blue) and resulting shapes.
# As we see,
# * A1 combines some stretching and rotation;
# * A2 projects the first transformation on the $XY$ plane;
# * A3 projects the first (or the second) transformation on a line.
# 
# Now, let's derive the properties of each matrix $A_i$.
# 
# $$
# rg(A_1) = 3;\hspace{3mm}rg(A_2) = 2;\hspace{3mm}rg(A_3) = 1
# $$
# 
# It follows from the Main theorem of Linear Algebra that
# 
# $$
# rg(ker(A_1)) = 0;\hspace{3mm}rg(ker(A_2)) = 1;\hspace{3mm}rg(ker(A_3)) = 2
# $$
# 
# That means, 
# * $A_1$ maps $\mathbb{R}^3$ to $\mathbb{R}^3$
# * $A_2$ maps $\mathbb{R}^3$ to $\mathbb{R}^2$
# * $A_3$ maps $\mathbb{R}^3$ to $\mathbb{R}^1$
# 
# That corresponds to our findings in the previous section.
# 
# Let us find the four fundamental subspaces for each $A_i$ operator:
# 
# * $A_1$ is a full-rank matrix, that means nullspace is empty. 
# 
# The same applies to $A_1^T$. $A_1$ column space is $
# \begin{bmatrix}
# 1\\
# 0\\
# 0
# \end{bmatrix}
# ,
# \begin{bmatrix}
# 0\\
# 2\\
# 0
# \end{bmatrix},
# \begin{bmatrix}
# 0\\
# 0\\
# 3
# \end{bmatrix}
# $, as well as for $A_1^T$ (row space of $A_1$).
# 
# * for $A_2$ we solve a simple set of linear equations and receive the following result:
# 
# $$Null(A_2) = span(
# \begin{bmatrix}
# 0\\
# 1\\
# -1
# \end{bmatrix}
# )
# ; \hspace{3mm} Null(A_2^T) = span(
# \begin{bmatrix}
# 0\\
# 1\\
# -1
# \end{bmatrix}
# ).
# $$
# 
# The column space of $A_2$ is
# $
# \begin{bmatrix}
# 1\\
# 0\\
# 0
# \end{bmatrix},
# \begin{bmatrix}
# 0\\
# 2\\
# 0
# \end{bmatrix}
# $, the row space is the same.
# * for $A_3$, we have two-dimensional null space: $Null(A_3) = span(
# \begin{bmatrix}
# 1\\
# 1\\
# -1
# \end{bmatrix},
# \begin{bmatrix}
# 1\\
# -1\\
# 1
# \end{bmatrix}
# )$. It coincides with the nullspace of $A_3^T$.
# 
# The column space of $A_3$ is $span(
# \begin{bmatrix}
# 1\\
# 1\\
# 0
# \end{bmatrix})$, as well as the row space. This result is easily observed on the visualisation: the result of the 3rd transformation lies on the $y = x$ line on the $XY$ plane.

# In[14]:


def print_eig(A, number):
    val, vec = np.linalg.eig(A)
    print("Eigenvalues and eigenvectors for matrix A{}:".format(number))
    print(val)
    print(vec)
    print()
    
#print_eig(A1, 1)
#print_eig(A2, 2)
#print_eig(A3, 3)


# **On transforming a cube into a tetrahedron:**
# Linear transformation implies that we can describe it only acting on basis vectors.
# That means, I suppose there is no such $3\times 3$ matrix to perform this operation.

# #### Problem 5
# 
# 
# For matrix
# $A=
# \begin{bmatrix}
# 2 & 1\\
# 1 & 2
# \end{bmatrix}
# $
# determine which unit vector $x_M$ is stretched the most and which $x_m$ the least and by how much. That is, find $x$ such that $y=Ax$ has the largest (or smallest) possible Euclidian length. You can do this by calculus methods, e.g. using Lagrange multipliers. Relate your findings to eigenvalues and eigenvectors of $A$.
# 
# 
# ##### Solution
# 
# Firstly, we SVD the matrix $A$:

# In[13]:


A = np.matrix([
    [2, 1],
    [1, 2]
])

u, s, v = np.linalg.svd(A)
print('U:\n', u)
print('E:', s)
print('V*:\n', v)


# We see that the matricies $U$ and $V*$ may be rewritten as a combination of the following matricies:
# 
# $$
# U = V^* = 
# \begin{bmatrix}
# -\frac{\sqrt{2}}{2} & -\frac{\sqrt{2}}{2}\\
# -\frac{\sqrt{2}}{2} & \frac{\sqrt{2}}{2}
# \end{bmatrix}
# =
# \frac{\sqrt{2}}{2}
# \begin{bmatrix}
# -1 & -1\\
# -1 & 1
# \end{bmatrix}
# =
# \frac{\sqrt{2}}{2}
# \begin{bmatrix}
# 1 & -1\\
# 1 & 1
# \end{bmatrix}
# \times
# \begin{bmatrix}
# -1 & 0\\
# 0 & 1
# \end{bmatrix}
# =\\
# =cR\times M,
# $$
# 
# where $c = \frac{\sqrt{2}}{2}$, $cR$ is a rotation matrix with $\varphi = \frac{1}{4}\pi$, $M$ is a mirror operator that mirrors about $X$ axis.
# 
# The matrix $\Sigma$ has diagonal elements $\begin{bmatrix} 3 & 1 \end{bmatrix}$ and represents stretching times 3 along $X$ axis.

# In[12]:


M = np.matrix([
    [-1, 0],
    [0, 1]
])
c = np.sqrt(2)/2
R = np.matrix([
    [1, -1],
    [1, 1]
])

c*R*M

c*R


# After the stretch has been completed, the matrix $U$ does the reverse transformation: mirrors the $X'$ axis and rotates $\frac{1}{4}\pi$ counterclockwise.
# 
# Now we can say that the most stretched vector $x_M$ will be the
# $
# \frac{\sqrt{2}}{2}
# \begin{bmatrix}
# 1\\
# 1
# \end{bmatrix}
# $ (or it's mirrored counterpart)
# unit vector, which during the transformations is aligned along the direction of the stretch operation and it's Euclidian length will be $3$, and the least stretched vector $x_m$ is the 
# $
# \frac{\sqrt{2}}{2}
# \begin{bmatrix}
# 1\\
# -1
# \end{bmatrix}
# $ (or it's mirrored counterpart), which lies perpendicular to the axis of the stretch operation. It's Euclidian length remains $1$.
# 
# Now let's look at the Eigenvalues and Eigenvectors of the matrix (the values and the vectors can be easily calculated by hand):

# In[11]:


val, vec = np.linalg.eig(A)
print('Eigenvalues: ', val)
print('Eigenvectors:')
print(vec)


# The result coincides with the result derived from SV decomposition: the
# $
# \frac{\sqrt{2}}{2}
# \begin{bmatrix}
# 1\\
# 1
# \end{bmatrix}
# $
# vector is stretched 3 times, the
# $
# \frac{\sqrt{2}}{2}
# \begin{bmatrix}
# 1\\
# -1
# \end{bmatrix}
# $
# vector remains the same.

# #### Problem 6
# 
# 
# Find eigenvalues and eigenvectors of the following matrices:  
# (a)
# $A_1=
# \begin{bmatrix}
# 0 & 1\\
# -1 & 0
# \end{bmatrix}
# $
# . If $x$ is any real vector, how is $y=A_1x$ related to $x$ geometrically?  
# 
# (b)
# $A_2=
# \begin{bmatrix}
# 1 & 1 & 0\\
# 0 & 1 & 1\\
# 0 & 0 & 1
# \end{bmatrix}
# $
# . What is the rank of $A_2$? How many eigenvectors are there?
# 
# 
# ##### Solution
# 
# The calculation of eigenvalues and eigenvectors may be performed by hand easily, as well as with *NumPy* package.
# 
# We start with matrix $A_1$:

# In[7]:


A1 = np.matrix([
    [0, 1],
    [-1, 0]
])

A2 = np.matrix([
    [1, 1, 0],
    [0, 1, 1],
    [0, 0, 1]
])

val1, vec1 = np.linalg.eig(A1)
val2, vec2 = np.linalg.eig(A2)

print_eig(A1, 1)


# Instantly from one glance at the matrix $A_1$, as well as by seeing the result of eigen_operations, we derive that the matrix conducts a rotation. There are no real eigenvectors, obviously, as there are no vectors $x$ that would be collinearly translated into another vectors $y$.

# Now, let's analyze the matrix $A_2$. It is obvious, that $rg(A_2) = 1$, because we can perform simple row operations ($r_2 = r_2 - r_3; \hspace{3mm} r_1 = r_1 - (r_2 - r_3)$) to get the resulting matrix $
# \begin{bmatrix}
# 1 & 0 & 0\\
# 0 & 0 & 0\\
# 0 & 0 & 0
# \end{bmatrix}
# $.

# In[10]:


print_eig(A2, 2)


# There is one eigenvalue $\lambda = 1$ of multiplicity 3, and only one eigenvector
# $h = 
# \begin{bmatrix}
# 1\\
# 0\\
# 0
# \end{bmatrix}
# $
# This matrix is a linear operator that projects on $YZ$ plane.

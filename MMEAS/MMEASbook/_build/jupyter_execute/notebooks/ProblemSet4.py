#!/usr/bin/env python
# coding: utf-8

# # Problem Set 4 

# ## Problem 1
# 
# Explain:
# 
# **(a)** why $A^TA$ is not singular when matrix $A$ has independent columns;  
# **(b)** why $A$ and $A^TA$ have the same nullspace
# 
# ### Solution
# 
# #### (a)
# 
# $A$ has independent columns $\Rightarrow$ it is full-rank, because column rank equals row rank. Then it has an inverse matrix $A^{-1}$ and:
# 
# $$
# A^TA = M,\\
# A^TAA^{-1} = MA^{-1},\\
# A^T = MA^{-1}
# $$
# 
# Because $A^T$ is a full-rank matrix ($rk(A^T) = rk(A)$), $M$ also has to be a full-rank matrix, thus, not singular.
# 
# #### (b)
# 
# Nullspace is a space spanned by all vectors $x$ such that $Ax = 0$. Let $x$ be any vector of that space $N(A)$. Then:
# 
# $$
# \forall x \in N(A) \rightarrow A^TAx = A^T(Ax) = A^T0 = 0.
# $$
# 
# We've proven that $N(A) \subset N(A^TA)$. To prove that $N(A^TA) \subset N(A)$, we do the following. Let $A^TAy = 0$, but $Ay \neq 0$. Then:
# 
# $$
# A^TAy = 0; \hspace{3mm} Ay = b \neq 0,\\
# A^Tb = 0 \Rightarrow b \in N(A^T).
# $$
# 
# But $N(A^T) \perp R(A^T) = C(A)$. This means that $b$ is not from column space of $A$, which is a contradiction, as $b = Ay$. Thus $b = Ay = 0$ and $N(A^TA) \subset N(A)$. Uniting the results, we obtain $N(A) = N(A^TA)$.

# ## Problem 2
# 
# A plane in $\mathbb{R}^3$ is given by the equation $x_1 - 2x_2 + x_3 = 0$.
# 
# **(a)** Identify two orthonormal vectors $u_1$ and $u_2$ that span the same plane.
# 
# **(b)** Find a projector matrix P that projects any vector from $\mathbb{R}^3$ to the plane and a projector $P_{\perp}$ that projects any vector to the direction normal to the plane.
# 
# **(c)** Using these two projectors find the unit normal to the plane and verify that it agrees with a normal found by calculus methods (that use the gradient).
# 
# ### Solution
# 
# #### (a)
# 
# We find any two lines belonging to the plane, vectorizethem, then perform orthogonalization and normalize the vectors.
# 
# The first vector $\hat{u}_1$ we can obtain by plugging $x_1 = 0$ into the plane equation. Then,
# 
# $$
# -2x_2 + x_3 = 0; \hspace{3mm} x_3 = 2x_2
# $$
# 
# and
# 
# $$
# \hat{u}_1 = \frac{1}{\sqrt{5}}\begin{bmatrix}0 & 1 & 2\end{bmatrix}^T.
# $$
# 
# The second $\hat{u}_2$ we find by plugging $x2 = 0$:
# 
# $$
# x_1 + x_3 = 0; \hspace{3mm} x_1 = -x_3
# $$
# 
# and
# 
# $$
# \hat{u}_2 = \frac{1}{\sqrt{2}}\begin{bmatrix}1 & 0 & -1\end{bmatrix}^T.
# $$
# 
# The orthogonalization:
# 
# $$
# u_2 = \hat{u}_2 - proj_{u_1}(u_2) =
# \hat{u}_2 - \hat{u}_1\frac{(\hat{u}_1, \hat{u}_2)}{|\hat{u}_1||\hat{u}_1|}
# $$
# 
# By this operations we receive:
# 
# $$
# u_1 = \hat{u}_1,\\
# u_2 = \frac{1}{\sqrt{30}}\begin{bmatrix}5 & 2 & -1\end{bmatrix}^T
# $$
# 
# #### (b)
# 
# In our case, as $u_1$ and $u_2$ are orthonormal, plane projector $P$ can be built as:
# 
# $$
# P = UU^T, \\
# U = \begin{bmatrix}u_1 & u_2\end{bmatrix}
# $$
# 
# We receive the following matrix:
# 
# $$
# P = 
# \begin{bmatrix}
# \frac{5}{6} & \frac{1}{3} & - \frac{1}{6}\\
# \frac{1}{3} & \frac{1}{3} & \frac{1}{3}\\
# - \frac{1}{6} & \frac{1}{3} & \frac{5}{6}
# \end{bmatrix}
# $$
# 
# Now, to obtain $P_{\perp}$, which is basically $nn^T$, where $n$ is a vector normal to the plane, we find the vector, whose coordinates are the coefficients in the equation of the plane:
# 
# ```{note}
# This is the gradient method
# ```
# 
# $$
# n = \frac{1}{\sqrt{6}}\begin{bmatrix}1 & -2 & 1\end{bmatrix} 
# $$
# 
# and put in into the matrix:
# 
# $$
# P_{\perp} = nn^T = 
# 
# \begin{bmatrix}
# \frac{1}{6} & - \frac{1}{3} & \frac{1}{6}\\
# - \frac{1}{3} & \frac{2}{3} & - \frac{1}{3}\\
# \frac{1}{6} & - \frac{1}{3} & \frac{1}{6}
# \end{bmatrix}
# $$
# 
# #### (c)
# 
# ```{note}
# As long as I've done the task in the wrong sequence, here is projector $P_{\perp}$ obtained by other means: $P_{\perp}$ projects to the nullspace of the $P$. So we find the vector that spans the nullspace of $P$ and make a projector out of it.
# ```
# $$
# N(P) = \begin{bmatrix}-1 & 2 & -1\end{bmatrix}^T
# $$
# 
# Well, from this point it is obvious, because the steps are the same.
# 
# ```{note}
# Below is the code for all necessary computations that were avoided being done by hand.
# ```

# 

# In[1]:


import sympy as sp

############ Finding u_2

u1h = sp.Matrix([0, 1, 2])*1/sp.sqrt(5)
u2h = sp.Matrix([1, 0, -1])*1/sp.sqrt(2)

u2 = u2h - u1h*u1h.dot(u2h)
u2 = u2.normalized()

u2


# In[2]:


############ Finding P

u1 = u1h

U = u1.row_join(u2)

P = U*U.T
P


# In[3]:


############ Finding P_perp

n = sp.Matrix([1, -2, 1]).normalized()

Pp = n*n.T
Pp


# In[4]:


############ P_perp by the means of the nullspace

a = sp.Matrix(P.nullspace())
a = a.normalized()
a*a.T


# ## Problem 3
# 
# Let $M = span\{v_1, v_2\}$, where $v_1 = \begin{bmatrix}1 & 0 & 1 & 1\end{bmatrix}^T$, $v_2 = \begin{bmatrix}1 & -1 & 0 & -1\end{bmatrix}^T$.  
# 
# **(a)** Find the orthogonal projector $P_M$ on $M$.  
# 
# **(b)** Find the kernel (nullspace) and range (column space) of $P_M$.
# 
# **(c)** Find $x \in M$ which is closest, in 2-norm, to the vector $a = \begin{bmatrix}1 & -1 & 1 & -1\end{bmatrix}^T$.

# ### Solution
# 
# #### (a)

# In[5]:


v1 = sp.Matrix([1, 0, 1, 1])
v2 = sp.Matrix([1, -1, 0, -1])

v1.dot(v2)


# $v_1 \perp v_2$, thus, we only have to normalize them and construct a projector as $P_M = QQ^T$:

# In[6]:


########## Constructing matrix Q from v_1 and v_2

v1 = v1.normalized()
v2 = v2.normalized()

M = v1.row_join(v2)
M


# In[7]:


########### Constructing projector

Pm = M*M.T
Pm


# #### (b)
# 
# Following the standard procedure of finding nullspace and column space (asking sympy to do this for us):

# In[8]:


######## Null space:
N = Pm.nullspace()
N[0].row_join(N[1])


# In[9]:


######### Column space:
C = Pm.columnspace()
C[0].row_join(C[1])


# #### (c)
# 
#  For this we need to project vector $a$ onto the $M$. This projection would be the closest to $a$ vector $x \in M$ in terms of 2-norm. So,
# 
#  $$
#  P_ma = b \in M
#  $$

# In[10]:


a = sp.Matrix([1, -1, 1, -1])
b = Pm*a
b


# Let's check the error norm:

# In[11]:


(a - b).norm()


# ## Problem 4
# 
# **(a)**
# 
# Using the determinant test, find $c$ and $d$ that make the following matrices positive definite:  
# 
# $$
# A = 
# \begin{bmatrix}
# c & 1 & 1\\
# 1 & c & 1\\
# 1 & 1 & c
# \end{bmatrix}
# , \hspace{3mm}
# B = 
# \begin{bmatrix}
# 1 & 2 & 3\\
# 2 & d & 4\\
# 3 & 4 & 5
# \end{bmatrix}
# $$
# 
# **(b)** A positive definite matrix cannot have a zero (or a negative number) on its main diagonal. Show that the matrix  
# 
# $$
# A = 
# \begin{bmatrix}
# 4 & 1 & 1\\
# 1 & 0 & 2\\
# 1 & 2 & 5
# \end{bmatrix}
# $$
# 
# is not positive by finding $x$ such shat $x^TAx \leq 0$.

# ### Solution
# 
# #### (a)
# 
# Both $A$ and $B$ are symmetric and quadratic. We can use Silvester's criteria:

# In[12]:


c, d = sp.symbols('c, d')
from sympy.solvers.inequalities import solve_poly_inequality
from sympy import Poly

A = sp.Matrix([
    [c, 1, 1],
    [1, c, 1],
    [1, 1, c]
])

B = sp.Matrix([
    [1, 2, 3],
    [2, d, 4],
    [3, 4, 5]
])

Inequalities_A = []
for i in range(0, max(A.shape)):
    det = A[0:i+1, 0:i+1].det()
    Inequalities_A.append(det)

Solutions_A = []

for ineq in Inequalities_A:
    Solutions_A.append(solve_poly_inequality(Poly(ineq), '>'))

Solutions_A


# We found the conditions for all minors to be positive. Now let's unite the results and find $c$:

# In[13]:


Unions_A = [sp.sets.Union(*sol) for sol in Solutions_A]
Intersection_A = sp.sets.Intersection(*Unions_A)
Intersection_A


# With any $c > 1$ matrix $A$ is positive-definite.
# 
# Now let's do the same for matrix $B$:

# In[14]:


Inequalities_B = []

for i in range(0, max(B.shape)):
    det = B[0:i+1, 0:i+1].det()
    Inequalities_B.append(det)

Inequalities_B


# As can be proven with simple calculations by hand, we have non-intersecting sets of possible $d$. Thus, matrix $B$ is not positive-definite with any $d$.
# 
# #### (b)
# 
# 

# In[15]:


from sympy import abc
x1, x2, x3 = abc.symbols('x1, x2, x3')

x = sp.Matrix([x1, x2, x3])

A = sp.Matrix([
    [4, 1, 1],
    [1, 0, 2],
    [1, 2, 5]
])

b = x.T*A*x
b[0]


# It is obvious that plugging $x_1 = 0; x_3 = 0$ into the result will give us $0$ with any $x_2$. Thus proven.
# 
# Also, only $x_2$ has no quadratic term, thus, we can decrease the product by decreasing the term $x_2$ with fixed $x_1$ and $x_3$:

# In[16]:


subs = {x1: 1, x2: -10, x3: 1}
b = b.subs(subs)
b


# ## Problem 5
# 
# Matrix $A = \begin{bmatrix}1&1&0\\2&3&1\\1&1&4\end{bmatrix}$ is positive definite. Explain why and determine the minimum value of $z = x^TAx + 2b^Tx + 1$, where $x = \begin{bmatrix}x_1 & x_2 & x_3\end{bmatrix}^T$ and $b^T = \begin{bmatrix}1 & -2 & 1\end{bmatrix}$

# ### Solution

# ```{warning}
# To be clear, I don't exactly understand the meaning "positive-definite" applied to a non-symmetric matrix. But okay...
# ```

# In[17]:


A = sp.Matrix([
    [1, 1, 0],
    [2, 3, 1],
    [1, 1, 4]
])

for i in range(0, max(A.shape)):
    string = '{} minor is: '.format(i+1)
    print(string, A[0:i+1, 0:i+1].det())


# So it is positive (in some way, e.g. [Totally Positive Matrix](https://en.wikipedia.org/wiki/Totally_positive_matrix)) due to positivity of its minors.

# In[18]:


from sympy import functions
x1, x2, x3 = abc.symbols('x1, x2, x3')

x = sp.Matrix([x1, x2, x3])
b = sp.Matrix([1, -2, 1])

z = x.T*A*x + 2*b.T*x + 1*sp.Matrix([1])
f = z[0]
f_difs = []

for x_ in x:
    f_difs.append(f.diff(x_))

f_difs = sp.Matrix(f_difs)
sols = sp.solve(f_difs)
sols


# We've got the $x$ which corresponds to the minimal value of $z$. Let's find it:

# In[19]:


z = z.subs(sols)
z[0]


# ```{note}
# This was the straightforward solution. Also we may perform some matrix manipulations first.
# ```
# 
# 
# $$
# 
# z = x^TAx + 2b^Tx + 1 = x^TA_+x + 2b^Tx + 1 = \frac{1}{2}x^T(A + A^T)x + 2b^Tx + 1;\\
# \frac{dz}{dx} = (A + A^T)x + 2b = 0,\\
# 
# (A + A^T)x = -2b \hspace{3mm}\Rightarrow\hspace{3mm} x = -2(A + A^T)^{-1}b.
# 
# 
# $$
# 
# From this point it is just a matter of calculations:

# In[20]:


y = -2*(A + A.T).inv()*b
y


# The same result.

# ## Problem 6
# 
# Explain these inequalities from the definitions of the norms:
# 
# $$
# ||ABx|| \leq ||A||||Bx|| \leq ||A||||B||||x||,
# $$
# 
# and deduce that $||AB|| \leq ||A||||B||$.

# ### Solution
# 
# For a given $x$:
# 
# $$
# 
# 
# ||ABx||_2^2 = (ABx)^T(ABx) = x^TB^TA^TABx,\\
# ||Bx||_2^2 = (Bx)^T(Bx) = x^TB^TBx,\\
# ||x||_2^2 = x^Tx,\\
# ||A||_2^2 = \max\limits_{x \in \mathbb{R}^n}\Bigg\{\frac{(Ax)^T(Ax)}{x^Tx}\Bigg\} = \max\limits_{x \in \mathbb{R}^n}\Bigg\{\frac{x^TA^TAx}{x^Tx}\Bigg\},\\
# ||B||_2^2 = \max\limits_{x \in \mathbb{R}^n}\Bigg\{\frac{(Bx)^T(Bx)}{x^Tx}\Bigg\} = \max\limits_{x \in \mathbb{R}^n}\Bigg\{\frac{x^TB^TBx}{x^Tx}\Bigg\}.
# 
# 
# $$
# 
# 
# Thus, we may rewrite some inequalities:
# 
# $$
# 
# 
# ||ABx|| = x^TB^TA^TABx \frac{x^TB^TBx}{x^TB^TBx} = \frac{x^tB^TA^TABx}{x^TB^TBx} x^TB^TBx =\\
# \hspace{3mm}\\
# = \frac{y^TA^TAy}{y^Ty} x^TB^TBx \leq \max\limits_{y \in \mathbb{R}^n}\Bigg\{\frac{y^TA^TAy}{y^Ty} \Bigg \} ||Bx|| = ||A||||Bx||;
# 
# \hspace{3mm}\\
# 
# ||Bx|| = x^TB^TBx = x^TB^TBx \frac{x^Tx}{x^Tx} = \frac{x^TB^TBx}{x^Tx} x^Tx \leq \max\limits_{z \in \mathbb{R}^n}\Bigg\{ \frac{z^TB^TBz}{z^Tz} \Bigg \} x^Tx = ||B||||x||
# 
# 
# $$
# 
# So, we've proven that $||ABx|| \leq ||A||||Bx||$ and $||Bx|| \leq ||B||||x||$, $\Rightarrow ||ABx|| \leq ||A||||Bx|| \leq ||A||||B||||x||$.
# 
# The first line we can use to prove that
# 
# $$
# 
# ||AB|| = \max\limits_{x \in \mathbb{R}^n}\frac{||ABx||}{||x||} = \max\limits_{x \in \mathbb{R}^n}\frac{||ABx||}{||x||} \frac{||Bx||}{||Bx||} =\\
# \hspace{3mm}\\
#  \max\limits_{x \in \mathbb{R}^n}\frac{||ABx||}{||Bx||}\frac{||Bx||}{||x||} \leq \max\limits_{x \in \mathbb{R}^n}\frac{||ABx||}{||Bx||}\max\limits_{x \in \mathbb{R}^n}\frac{||Bx||}{||x||} = ||A||||B||.
# 
# $$

# ## Problem 7
# 
# Compute by hand the norms and condition numbers of the following matrices:
# 
# $$
# A_1 =
# \begin{bmatrix}
# 2 & 1\\
# 1 & 2
# \end{bmatrix}
# ,\hspace{3mm}
# A_2 =
# \begin{bmatrix}
# 1 & 1\\
# -1 & 1
# \end{bmatrix}
# $$

# ### Solution
# 
# $A_1$ is symmetric, in this case its norm is the largest eigenvalue. Thus,
# 
# $$
# 
# 
# (2-\lambda)^2 - 1 = 0,\\
# 
# \lambda_1 = 3, \hspace{3mm} \lambda_2 = 1,\\
# 
# ||A_1|| = \max\limits_{i} \big \{ \lambda_i \big \} = 3,\\
# \kappa_1 = \frac{\max\limits_{i} \big \{ \lambda_i \big \}}{\min\limits_{i} \big \{ \lambda_i \big \}} = 3
# 
# 
# $$
# 
# 

# A_2 is skew-symmetric, for non-symmetric matrices the norm equals to the greatest singular value:

# In[21]:


A2 = sp.Matrix([
    [1, 1],
    [-1, 1]
])

A2.singular_value_decomposition()[1]


# $$
# 
# \sigma_1 = \sigma_2 = \sqrt{2},\\
# 
# ||A_2|| = \max\limits_{i} \big \{ \sigma_i \big \} = \sqrt{2},\\
# \kappa_2 = \frac{\max\limits_{i} \big \{ \sigma_i \big \}}{\min\limits_{i} \big \{ \sigma_i \big \}} = 1
# 
# $$

# $$
# 

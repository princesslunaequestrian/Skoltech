#!/usr/bin/env python
# coding: utf-8

# # Problem Set 10

# ## Problem 1

# Solve the initial value problem for the advection equation
# 
# $$
# u_t + (1-t)u_x = 0, \hspace{3mm} t>0, \hspace{3mm}x \in \mathbb{R}\\
# u(x, 0) = \frac{1}{1+x^2}.
# $$
# 
# Plot the characteristic curves as well as the solution $u(x, t)$ at several different times.

# ### Solution

# ```{image} ../attachments/PS10/ps10_1.jpg
# :alt: Problem 1
# :width: 400px
# :align: center
# ```

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# In[2]:


def fx(t, c):
    return t - t**2/2 + c

t = np.arange(0, 4, 0.1)
for c in range(-4, 4, 2):
    x = fx(t, c)
    plt.plot(x, t)

plt.title('Characteristic curves')
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.grid()
plt.legend(tuple(["$x_0 = {}$".format(i) for i in range(-4, 4, 2)]))


# In[3]:


def u(x, t):
    res = 1/(x - t + t**2/2)**2
    if (res >= 100):
        return 0
    else:
        return res


x = np.arange(-1, 1, 0.001)
t = np.arange(0.5, 1, 0.001)

X, T = np.meshgrid(x, t)
ni, nj = X.shape
U = np.zeros(X.shape)

for i in range(ni):
    for j in range(nj):
        U[i][j] = u(X[i][j], T[i][j])

ax = plt.subplot(projection='3d')
ax.plot_surface(X, T, U, cmap=cm.coolwarm)
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.title('Function plot')
ax.set_zlabel('$u(x,t)$')
plt.gcf().set_size_inches(10, 10)
ax.set_zlim((0, 100))


# ```{warning}
# Function is discontinuous at the points where the graph is weird
# ```

# ## Problem 2
# ### Solution
# ```{image} ../attachments/PS10/ps10_2.jpg
# :alt: Problem 1
# :width: 400px
# :align: center
# ```

# In[4]:


def u(x, t):
    if (x >= t):
        return x*t
    else:
        return x*t + t - x

x = np.arange(0, 3, 0.01)
t = np.arange(0, 3, 0.01)

X, T = np.meshgrid(x, t)
ni, nj = X.shape
U = np.zeros(X.shape)

for i in range(ni):
    for j in range(nj):
        U[i][j] = u(X[i][j], T[i][j])

ax = plt.subplot(projection='3d')
ax.plot_surface(X, T, U, cmap=cm.coolwarm)
plt.xlabel('$x$')
plt.ylabel('$t$')
plt.title('Function plot')
ax.set_zlabel('$u(x,t)$')
plt.gcf().set_size_inches(10, 10)


# ## Problem 3

# ### Solution

# ```{image} ../attachments/PS10/ps10_3.jpg
# :alt: Problem 1
# :width: 400px
# :align: center
# ```
# ```{image} ../attachments/PS10/ps10_4.jpg
# :alt: Problem 1
# :width: 400px
# :align: center
# ```

# ## Problem 4

# ### Solution

# ```{image} ../attachments/PS10/ps10_5.jpg
# :alt: Problem 1
# :width: 400px
# :align: center
# ```

# ## Problem 5

# ### Solution

# ```{image} ../attachments/PS10/ps10_6.jpg
# :alt: Problem 1
# :width: 400px
# :align: center
# ```
# ```{image} ../attachments/PS10/ps10_7.jpg
# :alt: Problem 1
# :width: 400px
# :align: center
# ```

# 

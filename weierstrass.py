import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error as mse

plt.rc("axes.spines", top = False, right = False)
plt.rc("axes", titlelocation = "left")
sns.set_style("whitegrid")

np.random.seed(0)

def W(x, order = 50, f0 = 0, a = 0.3, b = 7):
    """Returns the Weierstrass function at x. 
    Value of b is the smallest for which W is pathological for all a <- [0, 1].
    From: https://scipython.com/blog/the-weierstrass-function/ """
    return f0 + sum(a**k * np.cos(b**k * np.pi * x) for k in range(order))

# plot function
x = np.linspace(-5, 5, 5000)
plt.plot(x, W(x), linewidth = 0.5)
plt.gca().set_aspect('equal')
plt.show()

# UAR 
x_uar  = np.random.choice(x, 25)
W_uar  = W(x_uar)
gp_uar = GaussianProcessRegressor(kernel = Matern(nu = 2.5) + WhiteKernel(noise_level = 1))
Wp_uar, vp_uar = gp_uar\
    .fit(x_uar.reshape(x_uar.shape[0], -1), W_uar)\
    .predict(x.reshape(x.shape[0], -1), return_std = True)

mse(Wp_uar, W(x))

plt.plot(x, W(x), linewidth = 0.5)
plt.plot(x, Wp_uar)
plt.fill_between(x, Wp_uar - vp_uar, Wp_uar + vp_uar, alpha = 0.3, color = sns.color_palette()[1])
plt.gca().set_aspect('equal')
plt.show()

# loss based adaptive sampling

# uncertainty based sampling
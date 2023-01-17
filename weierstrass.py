import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error as mse

plt.rc("axes.spines", top = False, right = False)
plt.rc("axes", titlelocation = "left")
sns.set_style("whitegrid")

a0, b0 = 0.3, 7
def W(x, order = 50, f0 = 0, a = a0, b = b0):
    """Returns the Weierstrass function at x. 
    Default value of b is the smallest for which W is pathological for all a <- [0, 1].
    From: https://scipython.com/blog/the-weierstrass-function/ """
    return f0 + sum(a**k * np.cos(b**k * np.pi * x) for k in range(order))

def gp(nu = 2.5, c = 1):
    return GaussianProcessRegressor(kernel = Matern(nu = nu) + WhiteKernel(noise_level = c))

def reshape(x):
    return x.reshape(x.shape[0], -1)

blue, orange, green, red, *_ = sns.color_palette()

# plot function
x = np.linspace(-5, 5, 5000)
plt.plot(x, W(x), linewidth = 0.5)
plt.gca().set_aspect('equal')
plt.gca().set(title = f"Weierstrass function ($a$ = {a0}, $b$ = {b0})")
plt.savefig("weierstrass.png", dpi = 300)
plt.show()

# UAR 
np.random.seed(0)
x_uar = np.random.choice(x, 25)
W_uar = W(x_uar)
Wp_uar, vp_uar = gp()\
    .fit(reshape(x_uar), W_uar)\
    .predict(reshape(x), return_std = True)

mse_uar = mse(Wp_uar, W(x))
print(mse_uar)

plt.plot(x, W(x), linewidth = 0.5)
plt.plot(x, Wp_uar)
plt.fill_between(x, Wp_uar - vp_uar, Wp_uar + vp_uar, alpha = 0.3, color = orange)
plt.plot(x_uar, np.zeros(25), '.', markersize = 5, color = orange)
plt.gca().set_aspect('equal')
plt.gca().set(title = f"UAR, 25 points (MSE = {mse_uar.round(2)})")
plt.savefig("uar.png", dpi = 300)
plt.show()

# adaptive sampling
x_grid = np.linspace(-5, 5, 100)

## loss based
np.random.seed(0)
gp_loss = gp()
i = 0
x_loss = np.random.choice(x_grid, 5)
W_loss = W(x_loss)
x_pool = np.array([x for x in x_grid if x not in x_loss])
gp_loss.fit(reshape(x_loss), W_loss)
(Wp_loss, vp_loss) = gp_loss.predict(reshape(x_pool), return_std = True)
(Wp_grid, vp_grid) = gp_loss.predict(reshape(x),      return_std = True)
mse_loss = mse(Wp_grid, W(x))

plt.plot(x, W(x), linewidth = 0.5)
plt.plot(x, Wp_grid)
plt.fill_between(x, Wp_grid - vp_grid, Wp_grid + vp_grid, alpha = 0.3, color = orange)
plt.plot(x_loss, np.zeros((i+1)*5), '.', markersize = 5, color = orange)
plt.gca().set_aspect('equal')
plt.gca().set(title = f"loss-adaptive, {(i+1)*5} points (MSE = {mse_loss.round(2)})")
plt.savefig(f"loss-adaptive-step{i}.png", dpi = 300)
plt.show()

for i in range(1, 5):
    x_batch = [t[0] for t in sorted(zip(x_pool, (Wp_loss - W(x_pool))**2), key = lambda t: t[1], reverse = True)[:5]]
    x_loss  = np.append(x_loss, x_batch)
    W_loss = W(x_loss)
    x_pool  = np.array([x for x in x_grid if x not in x_loss])
    gp_loss.fit(reshape(x_loss), W_loss)
    (Wp_loss, vp_loss) = gp_loss.predict(reshape(x_pool), return_std = True)
    (Wp_grid, vp_grid) = gp_loss.predict(reshape(x),      return_std = True)
    mse_loss = mse(Wp_grid, W(x))

    plt.plot(x, W(x), linewidth = 0.5)
    plt.plot(x, Wp_grid)
    plt.fill_between(x, Wp_grid - vp_grid, Wp_grid + vp_grid, alpha = 0.3, color = orange)
    plt.plot(x_loss, np.zeros((i+1)*5), '.', markersize = 5, color = orange)
    plt.gca().set_aspect('equal')
    plt.gca().set(title = f"loss-adaptive, {(i+1)*5} points (MSE = {mse_loss.round(2)})")
    plt.savefig(f"loss-adaptive-step{i}.png", dpi = 300)
    plt.show()

## uncertainty based
np.random.seed(0)
gp_uncertainty = gp()
i = 0
x_uncertainty = np.random.choice(x_grid, 5)
W_uncertainty = W(x_uncertainty)
x_pool = np.array([x for x in x_grid if x not in x_uncertainty])
gp_uncertainty.fit(reshape(x_uncertainty), W_uncertainty)
(Wp_uncertainty, vp_uncertainty) = gp_uncertainty.predict(reshape(x_pool), return_std = True)
(Wp_grid, vp_grid) = gp_uncertainty.predict(reshape(x),      return_std = True)
mse_uncertainty = mse(Wp_grid, W(x))

plt.plot(x, W(x), linewidth = 0.5)
plt.plot(x, Wp_grid)
plt.fill_between(x, Wp_grid - vp_grid, Wp_grid + vp_grid, alpha = 0.3, color = orange)
plt.plot(x_uncertainty, np.zeros((i+1)*5), '.', markersize = 5, color = orange)
plt.gca().set_aspect('equal')
plt.gca().set(title = f"uncertainty-adaptive, {(i+1)*5} points (MSE = {mse_uncertainty.round(2)})")
plt.savefig(f"uncertainty-adaptive-step{i}.png", dpi = 300)
plt.show()

for i in range(1, 5):
    x_batch = [t[0] for t in sorted(zip(x_pool, vp_uncertainty), key = lambda t: t[1], reverse = True)[:5]]
    x_uncertainty  = np.append(x_uncertainty, x_batch)
    W_uncertainty = W(x_uncertainty)
    x_pool  = np.array([x for x in x_grid if x not in x_uncertainty])
    gp_uncertainty.fit(reshape(x_uncertainty), W_uncertainty)
    (Wp_uncertainty, vp_uncertainty) = gp_uncertainty.predict(reshape(x_pool), return_std = True)
    (Wp_grid, vp_grid) = gp_uncertainty.predict(reshape(x),      return_std = True)
    mse_uncertainty = mse(Wp_grid, W(x))

    plt.plot(x, W(x), linewidth = 0.5)
    plt.plot(x, Wp_grid)
    plt.fill_between(x, Wp_grid - vp_grid, Wp_grid + vp_grid, alpha = 0.3, color = orange)
    plt.plot(x_uncertainty, np.zeros((i+1)*5), '.', markersize = 5, color = orange)
    plt.gca().set_aspect('equal')
    plt.gca().set(title = f"uncertainty-adaptive, {(i+1)*5} points (MSE = {mse_uncertainty.round(2)})")
    plt.savefig(f"uncertainty-adaptive-step{i}.png", dpi = 300)
    plt.show()
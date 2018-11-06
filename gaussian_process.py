import numpy as np
import matplotlib.pyplot as plt
import george

kernel = george.kernels.ExpSquaredKernel(1.)
gp = george.GP(kernel)

n = 100
x = np.linspace(0, n, 100)
ys = gp.sample(x, n)

for y in ys:
    plt.plot(x, y)
plt.show()

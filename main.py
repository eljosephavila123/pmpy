import pmpy 
import matplotlib.pyplot as plt
im=pmpy.perlin_noise.generate_fractal_noise_2d([100,100],[10,10],porosity=0.55)
plt.show(block=False)
plt.imshow(im)
plt.pause(1000)

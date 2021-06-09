import pmpy as pm
import porespy as ps
import scipy as sp

im = pm.perlin_noise.generate_fractal_noise_2d(
    (100, 100, 100), (1, 4, 4), 4, tileable=(True, False, False)
)

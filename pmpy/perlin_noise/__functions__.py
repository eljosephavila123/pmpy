import numpy as np
from scipy import special
import scipy as sp


def norm_to_uniform(im, scale=None):
    if scale is None:
        scale = [im.min(), im.max()]
    im = (im - np.mean(im)) / np.std(im)
    im = 1 / 2 * sp.special.erfc(-im / np.sqrt(2))
    im = (im - im.min()) / (im.max() - im.min())
    im = im * (scale[1] - scale[0]) + scale[0]
    return im


def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

def binary_medium(noise):
    shape_=noise.shape
    for x in range(shape_[0]):
        for y in range(shape_[1]):
            if noise[x][y]<-0.5:
                noise[x][y]=0
            elif noise[x][y]<0:
                noise[x][y]=0
            elif noise[x][y]<1.0:
                noise[x][y]=1
    return noise

def generate_fractal_noise_2d(shape, res, octaves=1, persistence=2, porosity=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
 
    noise = norm_to_uniform(noise, scale=[0, 1])
    return noise  > porosity







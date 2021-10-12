#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stick figure manifold


Created on Sun Aug 22 18:06:06 2021

@author: robert
"""

from __future__ import annotations

from typing import List, Callable

import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass, field

from numpy.random import sample

from flax import linen
from jax import random
from jax.tree_util import tree_flatten, tree_unflatten

from PIL import Image

class Limb(linen.Module):
    
    init_angle: float = 0
    init_length: float = 0.5

    zero: float = 0
    relative: bool = True
    attach_point: float = 0
    
    children: List[Limb] = field(default_factory=list)

    def setup(self):
        
        self.angle = self.param('angle',lambda key,a: a, self.init_angle)
        # self.length  = self.param('length',lambda key,a: a,  self.init_length)
        self.length = self.init_length
        
            
    def __call__(self,p,base_angle=0):
        
        angle = self.zero + self.angle
        if self.relative:
            angle += base_angle
        
        p = np.array(p)
        vec = np.array([np.cos(angle),np.sin(angle)])
        
        points = [(p,p+vec*self.length)]
        
        for child in self.children:
            r = p+vec*child.attach_point*self.length
            points += child(r,angle)
            
        return points
   

def plot_points(points,linewidth=4,get_array=True):
    
    fig = plt.figure(figsize=(1, 1), dpi=32)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([-0.8,0.8])
    ax.set_ylim([-0.8,0.8])

    fig.add_axes(ax)
                    
    for (p1,p2) in points:
        x = [p1[0],p2[0]]
        y = [p1[1],p2[1]]
        ax.plot(x,y,color='black',linewidth=linewidth)

    if get_array:
        fig.set_dpi(32)
        fig.canvas.draw()
        x = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        return x   

def make_man() -> List[Limb,Callable]:

    from numpy import pi

    calfl = Limb(init_angle=0.4,attach_point=1)
    thighl = Limb(zero=pi,init_angle=-0.2,children=[calfl])

    calfr = Limb(init_angle=0.4,attach_point=1)
    thighr = Limb(zero=pi,init_angle=-0.2,children=[calfr])

    forearml = Limb(init_angle=-0.5,attach_point=1,init_length=0.4)
    arml = Limb(zero=pi,init_angle=0.2,attach_point=0.9,init_length=0.4,children=[forearml])

    forearmr = Limb(init_angle=-0.5,attach_point=1,init_length=0.4)
    armr = Limb(zero=pi,init_angle=0.2,attach_point=0.9,init_length=0.4,children=[forearmr])

    torso = Limb(zero=pi/2,init_length=1,children=[thighl,arml,thighr,armr])

    key = random.PRNGKey(0)
    params = torso.init(key,[0,0])

    def sample_fun():

        nonlocal params

        leaves,treedef = tree_flatten(params)

        # mu = np.zeros(len(leaves))
        mu = [0.0,-0.2,0.4,0.2,-0.4,-0.2,0.4,0.2,-0.4]
        # var = np.ones(len(leaves))*0.1
        var = [0.03,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]

        cov = np.zeros((len(leaves),)*2)
        cov[1,2] = -0.03
        cov[3,4] = -0.03

        cov[1,5] = -0.06
        cov[3,6] = -0.06

        cov += cov.T
        cov += np.diag(var)

        leaves = np.random.multivariate_normal(mu,cov)
        params = tree_unflatten(treedef,leaves)

        return params

    return torso, sample_fun

from scipy.ndimage import gaussian_filter
def preprocess(img,sigma=1):
    img = img[:,:,0].astype(np.float32)/255
    img = gaussian_filter(img,sigma=sigma)
    img = np.repeat((img[:,:,None] * 255).astype(np.uint8),3,axis=2)
    return img

import pickle
def save_frame(img,params,fname):
    img = Image.fromarray(img)
    img.save(fname + ".png")

    with open(fname + ".pickle","wb") as f:
        pickle.dump(params,f)


from tqdm import tqdm

def build_dataset(name,num_samples=100,sigma=2,linewidth=4):

    torso, sample_fun = make_man()

    fname_template = f'{name}/img_{{}}'

    for ind in tqdm(range(num_samples)):

        params = sample_fun()
        points = torso.apply(params,[0,0])

        fname = fname_template.format(ind)

        img = plot_points(points,linewidth=linewidth)        
        img = preprocess(img,sigma=sigma)

        save_frame(img,params,fname)


    
if __name__ == "__main__":
    
    build_dataset(name='g',num_samples=20000,sigma=0.5,linewidth=5)

        
    
    
    
    
    
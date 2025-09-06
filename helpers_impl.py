#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 06:41:02 2023

@author: chi1
"""

import fire
import matplotlib.pyplot as plt
import torch
import torch.optim.swa_utils as swa_utils
import torchcde
import torchsde
from torch import nn
import copy
import matplotlib.pyplot as plt
from torch import profiler
from tqdm import tqdm
import numpy as np
from typing import Sequence
import pdb
from torchdiffeq import odeint
import control
import scipy

from typing import Union
import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import optax  # https://github.com/deepmind/optax
import jax
import lineax
import pynvml
import pdb
def get_gpu_memory():
    pynvml.nvmlInit()
    num_devices = pynvml.nvmlDeviceGetCount()
    memory_info = []
    
    for i in range(num_devices):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_info.append({
            'device': i,
            'total_memory': mem_info.total / (1024 ** 2),  # Convert bytes to MB
            'free_memory': mem_info.free / (1024 ** 2),    # Convert bytes to MB
            'used_memory': mem_info.used / (1024 ** 2)     # Convert bytes to MB
        })
    
    pynvml.nvmlShutdown()
    return memory_info
    
def lipswish(x):
    return 0.909 * jnn.silu(x)

def generate_system_canon(dim):
    """
    Generate a random controllable system in controllable canonical form.
    
    Args:
        n (int): The size of the state-space (number of states).
    
    Returns:
        A (numpy.ndarray): The state matrix in companion form (n x n).
        B (numpy.ndarray): The input matrix (n x 1).
    """

    # Step 1: Generate random coefficients for the characteristic polynomial of A
    # Random coefficients for the polynomial (a_0, a_1, ..., a_(n-1))
    polynomial_coefficients = np.random.normal(0, 1, dim)  # Coefficients in [-10, 10]
  
    # Step 2: Construct the companion matrix A
    A = np.zeros((dim, dim))
    np.fill_diagonal(A[:, 1:], 1)  # Fill the superdiagonal with 1s
    A[-1, :] = -polynomial_coefficients  # Last row contains -[a_0, a_1, ..., a_(n-1)]

    # Step 3: Construct the B matrix
    B = np.zeros((dim, 1))
    B[-1, 0] = 1  # Last element of B is 1

    # Step 4: Construct the C matrix
    C = np.zeros((1, dim))
    C[0,0] = 1
    
    if np.linalg.matrix_rank(control.ctrb(A, B)) == dim:
        print('system is controllable')

    if np.linalg.matrix_rank(control.obsv(A, C)) == dim:
        print('system is observable')
    
    return A, B

def generate_system_canon2(dim, num_inputs):
    """
    Generate a random controllable system in controllable canonical form with multiple inputs.
    
    Args:
        dim (int): The size of the state-space (number of states, n).
        num_inputs (int): The number of inputs (m).
        key (int): Random seed for reproducibility.
    
    Returns:
        A (jax.ndarray): The state matrix in companion form (n x n).
        B (jax.ndarray): The input matrix (n x m).
    """
    # Ensure num_inputs <= dim to maintain controllability
    if num_inputs > dim:
        raise ValueError("Number of inputs (m) must be less than or equal to state dimension (n).")

    # Set random seed for reproducibility
    

    # Step 1: Generate random coefficients for the characteristic polynomial of A
    polynomial_coefficients = np.random.normal(0, 1, dim)  # Random coefficients from normal distribution

    # Step 2: Construct the companion matrix A
    A = np.zeros((dim, dim))
    np.fill_diagonal(A[:, 1:], 1)  # Fill the superdiagonal with 1s
    A[-1, :] = -polynomial_coefficients  # Last row contains -[a_0, a_1, ..., a_(n-1)]

    # Step 3: Construct the B matrix for multiple inputs
    B = np.zeros((dim, num_inputs))
    # Set the last m rows of B to an m x m identity matrix
    B[-num_inputs:, :] = np.eye(num_inputs)

    # Step 4: Construct the C matrix (single output for observability check)
    C = np.zeros((1, dim))
    C[0, 0] = 1

    # Step 5: Check controllability
    if np.linalg.matrix_rank(control.ctrb(A, B)) == dim:
        print('System is controllable')
    else:
        print('System is NOT controllable')

    # Step 6: Check observability
    if np.linalg.matrix_rank(control.obsv(A, C)) == dim:
        print('System is observable')
    else:
        print('System is NOT observable')

    return A, B


def generate_system(dim):
    while True:
        A, B = np.random.randn(dim, dim), np.random.randn(dim, dim)
        A, B = A / np.max(np.abs(np.linalg.eigvals(A))), B / np.max(np.abs(np.linalg.eigvals(B)))
        if np.linalg.matrix_rank(control.ctrb(A, B)) == dim:
            return A, B

class VectorField2(eqx.Module):
   
    poly0: eqx.nn.MLP

    A: jax.Array
    B: jax.Array
    hidden_size: int
        
    def __init__(self, hidden_size, width_size, depth, A, B,  *, key, **kwargs):
        super().__init__(**kwargs)
        poly0_key, poly1_key = jr.split(key, 2)
        self.A = A 
        self.B = B
    
        self.poly0 = eqx.nn.MLP(
            in_size=1+hidden_size,
            out_size=B.shape[1], 
            width_size=width_size,
            depth= depth,
            key= poly0_key,
        )
        

        
        self.hidden_size=hidden_size
    def __call__(self, t, y,args):
     

        #jax.debug.breakpoint()
        #calculate intrinsic force
        f1=args[0] @ y[0:self.hidden_size]
     
        yt= jnp.concatenate([jnp.expand_dims(t, 0), y[0:self.hidden_size]])
    
        #calculate external foce
        tf=self.poly0(yt)

        f2=args[1] @ tf
         
        f=f1+f2
        fs=jnp.expand_dims(jnp.sum(tf**2),0)
        
        ff=jnp.concatenate([f,fs])
        
        
        return ff


class VectorField(eqx.Module):
   
    poly0: eqx.nn.MLP

    A: jax.Array
    B: jax.Array
    hidden_size: int
        
    def __init__(self, hidden_size, width_size, depth, A, B,  *, key, **kwargs):
        super().__init__(**kwargs)
        poly0_key, poly1_key = jr.split(key, 2)
        self.A = A 
        self.B = B
    
        self.poly0 = eqx.nn.MLP(
            in_size=1,
            out_size=B.shape[1], 
            width_size=width_size,
            depth= depth,
            key= poly0_key,
        )
        

        
        self.hidden_size=hidden_size
    def __call__(self, t, y,args):
     

        #jax.debug.breakpoint()
        #calculate intrinsic force
      
        f1=args[0] @ y[0:self.hidden_size]
     
        #calculate external foce
        tf=self.poly0(jnp.expand_dims(t, 0))

        f2=args[1] @ tf
         
        f=f1+f2
        fs=jnp.expand_dims(jnp.sum(tf**2),0)
        
        ff=jnp.concatenate([f,fs])
        
        
        return ff



    
class NeuralODEPlant(eqx.Module):
   
    vf: VectorField  # drift
 
    #vf2: VectorField2

    hidden_size: int

        
    def __init__(
        self,
        data_size,
        hidden_size,
        width_size,
        depth,
        A,
        B,
        *,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        vf_key, cvf_key = jr.split(key, 2)
 
        
     
        self.hidden_size=hidden_size
      
        

        
        self.vf = VectorField(hidden_size, width_size, depth, A, B, key=vf_key)
        #self.vf2 = VectorField2(hidden_size, width_size, depth, A, B, key=vf_key)
      
   

    def __call__(self, ts, y0, args):
        t0 = ts[0]
        t1 = ts[-1]
        # Very large dt0 for computational speed
       
        dt=(t1-t0)/ts.shape[0]
      
        #init = jr.normal(init_key, (self.initial_noise_size,))
 
        #drift = lambda t, y, args: -0*y
        vf = diffrax.ODETerm(self.vf)  # Drift term
        #vf = diffrax.ODETerm(self.vf2)
        #vf = diffrax.ODETerm(drift)
  
     
   
        # ReversibleHeun is a cheap choice of SDE solver. We could also use Euler etc.
        solver = diffrax.ReversibleHeun()
      
        #y0 = jnp.append(self.y0, jnp.zeros((1,1)))
       
        saveat = diffrax.SaveAt(ts=ts)
       
        sol = diffrax.diffeqsolve(vf, solver, t0, t1, dt0=dt, y0=y0, saveat=saveat,args=args)
       
        return sol.ys 
        
    
def laplacian_from_adjacency(adj_matrix):
    # Degree matrix (diagonal matrix with degree of nodes)
    degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
    # Laplacian matrix: L = D - A
    laplacian = degree_matrix - adj_matrix
    return laplacian


def criticalK(adjm, data_size, scale):
    L=laplacian_from_adjacency(adjm)
    #L=np.matmul(adjm,np.transpose(adjm))
    
    eigvals = scipy.linalg.eigh(L, eigvals_only=True)

    eigvals = np.sort(eigvals)

    # Get largest and second smallest
    second_smallest = eigvals[1]
    largest = eigvals[-1]
   
    K= data_size * (np.pi**2/4) * np.sqrt(data_size) * largest/ (scale*second_smallest**2)

    return K, L

def order_param(y, L):
    # y: shape (batch_size, N)
    # L: shape (N, N)

    ysin = jnp.sin(y)  # shape (batch_size, N)
    ycos = jnp.cos(y)
   
    # Expand L to shape (batch_size, N, N)
    batch_size = y.shape[0]
    N = y.shape[1]
    L = jnp.broadcast_to(L, (batch_size, N, N))

    # Compute quadratic forms
    quad_form_sin = jnp.einsum('bi,bij,bj->b', ysin, L, ysin)
    quad_form_cos = jnp.einsum('bi,bij,bj->b', ycos, L, ycos)
  
  
    # Compute order parameter
    r = jnp.ones((batch_size)) - (quad_form_sin + quad_form_cos) / (N ** 2)
   
    return r


def global_param(y, L):
    # y: shape [T, B, N]
    ysin = jnp.sin(y)
    ycos = jnp.cos(y)
    
    # Compute quadratic forms
    quad_form_sin = jnp.einsum('tbi,ij,tbj->tb', ysin, L, ysin)
    quad_form_cos = jnp.einsum('tbi,ij,tbj->tb', ycos, L, ycos)
    
    # Normalize and compute r
    N = y.shape[-1]
    quad_forms = (quad_form_sin + quad_form_cos) / (N ** 2)
    r = 1.0 - quad_forms
    return r




def objective(y, A):
  
    ymat = jnp.expand_dims(y, 1)   # shape (9, 1)
    ytr = jnp.tile(ymat, (1, y.shape[0]))
    
    tmat=jnp.transpose(ytr, (1,0))
    
    phasematrix=tmat-ytr

    forcematrix=jnp.sin(phasematrix)  
    forcematrix=jnp.square(forcematrix) * A
  
    
    return jnp.sum(forcematrix, axis=(0,1))/2

def objective_test(y, A):
  
    ymat = jnp.expand_dims(y, 2)   # shape (9, 1)
    ytr = jnp.tile(ymat, (1,1, y.shape[1]))
    #pdb.set_trace()
    tmat=jnp.transpose(ytr, (0,2,1))
    
    phasematrix=tmat-ytr

    forcematrix=jnp.sin(phasematrix)  
    forcematrix=jnp.square(forcematrix) * A
  
    
    return jnp.sum(forcematrix, axis=(1,2))/2


def sqr_adj(dim):
    
    dim=int(np.sqrt(dim))
    adj=np.zeros((dim*dim,dim*dim))
    
    for i in range(0, dim*dim):
        for j in range(0, dim*dim):

            if (j+1)==i and (i%dim != 0):
                adj[i,j]=1
            if (j-1)==i and (j%dim != 0):
                adj[i,j]=1
            if (j-dim)==i:
                adj[i,j]=1
            if (j+dim)==i:
                adj[i,j]=1
                
    return adj


def erdos_graph(dim, p):
   
    
    G=nx.erdos_renyi_graph(dim,p)
    A = nx.adjacency_matrix(G)

    return A.toarray()

def watts_graph(dim, k, p):

    G= nx.watts_strogatz_graph(dim,k,p)
    A= nx.adjacency_matrix(G)

    return A.toarray()

def adj_lattice(dim, graph, *args):
    
    if graph=="square":
     
        return sqr_adj(dim)
                
    elif graph == "erdos":

        return erdos_graph(dim, args[0])

    elif graph == "watts":

        return watts_graph(dim, args[1], args[0])


class VectorFieldKuramoto(eqx.Module):
   
    poly0: eqx.nn.MLP


    hidden_size: int
        
    def __init__(self, hidden_size, width_size, depth,  *, key, **kwargs):
        super().__init__(**kwargs)
        poly0_key, poly1_key = jr.split(key, 2)

       
        self.poly0 = eqx.nn.MLP(
            in_size=1,
            out_size=1, 
            width_size=width_size,
            depth= depth,
            key= poly0_key,
        )
        

        
        self.hidden_size=hidden_size
    def __call__(self, t, y,args):
     
        ymat=jnp.repeat(jnp.expand_dims(y[0:self.hidden_size],0),self.hidden_size,axis=0)
        
        tmat=jnp.transpose(ymat)
 
        phasematrix=tmat-ymat
        forcematrix= jnp.sin(phasematrix) * args[1]
        #jax.debug.print("🤯sinmatrix{sinmatrix} 🤯", sinmatrix=jnp.sin(phasematrix))
        #calculate intrinsic force
        freqs= args[0]
        #calculate external foce
      
        tf=self.poly0(jnp.expand_dims(t, 0))
        
        force=args[2]*tf*jnp.sum(forcematrix,axis=1)/self.hidden_size
        force+=freqs
        
     
        fs=jnp.expand_dims(jnp.mean(tf**2),0)
        
        ff=jnp.concatenate([force,fs])
        
      
        return ff

class VectorFieldKuramotoVec(eqx.Module):
   
    poly0: eqx.nn.MLP


    hidden_size: int
        
    def __init__(self, hidden_size, width_size, depth,  *, key, **kwargs):
        super().__init__(**kwargs)
        poly0_key, poly1_key = jr.split(key, 2)

       
        self.poly0 = eqx.nn.MLP(
            in_size=1,
            out_size=hidden_size, 
            width_size=width_size,
            depth= depth,
            key= poly0_key,
        )
        

        
        self.hidden_size=hidden_size
    def __call__(self, t, y,args):
     
        ymat=jnp.repeat(jnp.expand_dims(y[0:self.hidden_size],0),self.hidden_size,axis=0)
        
        tmat=jnp.transpose(ymat)
 
        phasematrix=tmat-ymat
        forcematrix= jnp.sin(phasematrix) * args[1]
        #jax.debug.print("🤯sinmatrix{sinmatrix} 🤯", sinmatrix=jnp.sin(phasematrix))
        #calculate intrinsic force
        freqs= args[0]
        #calculate external foce
      
        tf=self.poly0(jnp.expand_dims(t, 0))
        
        force=args[2]*tf*jnp.sum(forcematrix,axis=1)/self.hidden_size
        force+=freqs
        
     
        fs=jnp.expand_dims(jnp.mean(tf**2),0)
        
        ff=jnp.concatenate([force,fs])
        
      
        return ff


class VectorFieldKuramotoVecCL(eqx.Module):
   
    poly0: eqx.nn.MLP


    hidden_size: int
        
    def __init__(self, hidden_size, width_size, depth,  *, key, **kwargs):
        super().__init__(**kwargs)
        poly0_key, poly1_key = jr.split(key, 2)

       
        self.poly0 = eqx.nn.MLP(
            in_size=1+hidden_size,
            out_size=hidden_size, 
            width_size=width_size,
            depth= depth,
            key= poly0_key,
        )
        

        
        self.hidden_size=hidden_size
    def __call__(self, t, y,args):
     
        ymat=jnp.repeat(jnp.expand_dims(y[0:self.hidden_size],0),self.hidden_size,axis=0)
        
        tmat=jnp.transpose(ymat)
 
        phasematrix=tmat-ymat
        forcematrix= jnp.sin(phasematrix) * args[1]
        #jax.debug.print("🤯sinmatrix{sinmatrix} 🤯", sinmatrix=jnp.sin(phasematrix))
        #calculate intrinsic force
        freqs= args[0]
        #calculate external foce
    
        tf=self.poly0(jnp.concatenate([jnp.expand_dims(t, 0),y[0:self.hidden_size]]))
        
        force=args[2]*tf*jnp.sum(forcematrix,axis=1)/self.hidden_size
        force+=freqs
        
     
        fs=jnp.expand_dims(jnp.mean(tf**2),0)
        
        ff=jnp.concatenate([force,fs])
        
      
        return ff


class NeuralSDEKuramoto(eqx.Module):
   
    vf: VectorFieldKuramoto  # drift
    vf_vec: VectorFieldKuramotoVec
    vf_vec_cl: VectorFieldKuramotoVecCL

    strategy: str
    hidden_size: int      
 
  
    def __init__(
        self,
        data_size,

        hidden_size,
        width_size,
        depth,

        strategy,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        vf_key, cvf_key = jr.split(key, 2)

        
     
        self.hidden_size=hidden_size
        self.strategy= strategy
 
        self.vf = VectorFieldKuramoto(hidden_size, width_size, depth,  key=vf_key)
        self.vf_vec= VectorFieldKuramotoVec(hidden_size, width_size, depth,  key=vf_key)
        self.vf_vec_cl= VectorFieldKuramotoVecCL(hidden_size, width_size, depth,  key=vf_key)
        
    def __call__(self, ts, y0,  freqs, A, Kcrit):
        t0 = ts[0]
        t1 = ts[-1]
        # Very large dt0 for computational speed
       
        dt=(t1-t0)/ts.shape[0]
 
        #drift = lambda t, y, args: -0*y
        if self.strategy=="scalar":
            vf = diffrax.ODETerm(self.vf)  # Drift term
        elif self.strategy=="vector":
            vf= diffrax.ODETerm(self.vf_vec)
        else:
            vf=diffrax.ODETerm(self.vf_vec_cl)
        # ReversibleHeun is a cheap choice of SDE solver. We could also use Euler etc.
        solver = diffrax.ReversibleHeun()
    
        y0 = jnp.concatenate([y0, jnp.array([0.0])])
        saveat = diffrax.SaveAt(ts=ts)
        args = (freqs, A, Kcrit)
        sol = diffrax.diffeqsolve(vf, solver, t0, t1, dt0=dt, y0=y0, saveat=saveat, args=args,)
      
        return sol.ys    
    
    
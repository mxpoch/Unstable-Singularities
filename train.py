import haiku as hk
import jax 
import jax.numpy as jnp
import jax.random as jr
import optax
import numpy as np
# Custom high-accuracy Hilbert transform
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple
import chex


# Haiku-based NN used to learn profiles for Omega and U
# 3 hidden layers with tanh activation functions, and an ELU as the final activation
def nnet(z): 
    mlp = hk.Sequential([
        hk.Linear(20), jax.nn.tanh,
        hk.Linear(20), jax.nn.tanh,
        hk.Linear(20), jax.nn.tanh,
        hk.Linear(1), jax.nn.elu
    ])
    return mlp(z)

# defining the output function q for the DG equations
def q_DG(z, lambda_val):
    zs = z.reshape(1, -1)
    # evaluating the function
    nn_z = nnet(zs)
    nn_neg_z = nnet(-zs)
    q = ((nn_z - nn_neg_z) / 2)*(1+z**2)**(-1/(2*(1+lambda_val)))
    return jnp.squeeze(q)

# making q JAX-compatible
q_DG_jax = hk.transform(q_DG)
# autodiff gradients
dq_dz = jax.grad(q_DG_jax.apply, argnums=2)
d3q_dz3 = jax.grad(jax.grad(jax.grad(q_DG_jax.apply, argnums=2), argnums=2), argnums=2)

# vmappings
in_axes = (None, None, 0, None)
q_DG_vmap = jax.vmap(q_DG_jax.apply, in_axes=in_axes)
dq_dz_vmap = jax.vmap(dq_dz, in_axes=in_axes)
d3q_dz3_vmap = jax.vmap(d3q_dz3, in_axes=in_axes)

# 2nd order lagrange polynomials
def _simpson_weights(n_points):
    if n_points % 2 == 0:
        raise ValueError("Simpson's rule requires an odd number of grid points.")
    
    # [1, 4, 2, 4, ..., 2, 4, 1]
    weights = jnp.ones(n_points)
    weights = weights.at[1::2].set(4.0)
    weights = weights.at[2:-1:2].set(2.0)
    return weights

s_grid = jnp.linspace(-30,30,50001)
s_grid = s_grid.reshape(s_grid.shape[0],1)
L = s_grid[-1]

n_points = s_grid.shape[0]
h = (s_grid[-1] - s_grid[0]) / (n_points - 1)
weights = _simpson_weights(n_points)

def Hn(params, rng, z, lambda_val):
    omega_at_z = q_DG_jax.apply(params, rng, z, lambda_val)
    omega_on_grid = q_DG_vmap(params, rng, s_grid, lambda_val)

    # This integrand is now smooth at s=z
    integrand = (omega_on_grid - omega_at_z) / (s_grid[0].squeeze() - z)
    integral_part = (h / 3.0) * jnp.dot(weights, integrand)

    # P.V. integral of 1/(z-s) from -L to L is log(|(L-z)/(L+z)|)
    # Small epsilon for numerical stability if z == L or z == -L
    epsilon = 1e-10
    log_term = jnp.log(jnp.abs((L - z + epsilon) / (L + z + epsilon)))
    analytical_part = omega_at_z * log_term

    return ((integral_part + analytical_part) / jnp.pi).squeeze()

# Loss functions
def conditional_loss(Omega_p, rng, lambda_val):
      # normalization constant 
      g1 = (q_DG_jax.apply(Omega_p, rng, jnp.array([[0.5]]), lambda_val) + 0.05)**2
      # uniform sampling of points to decay at infinity
      bd_pts = jnp.concatenate([jr.uniform(rng, shape=(10,), minval=29, maxval=30), 
                              jr.uniform(rng, shape=(10,), minval=-30, maxval=-29)])
      bd_pts = bd_pts.reshape(bd_pts.shape[0],1) # reshaping for batch size
      g2 = 1/20*jnp.sum(q_DG_vmap(Omega_p, rng, bd_pts, lambda_val)**2)

      return 1/2*(g1+g2)

# first equation residue
def f1(Omega_p, U_p, rng, z, lambda_val):
      Omega_z = q_DG_jax.apply(Omega_p, rng, z, lambda_val)
      dOmega_dz = dq_dz(Omega_p, rng, z, lambda_val)
      U_z = q_DG_jax.apply(U_p, rng, z, lambda_val)
      dU_dz = dq_dz(U_p, rng, z, lambda_val)

      summation = (Omega_z + ((1+lambda_val)*jax.numpy.sinh(z) - U_z)
              *(1/jax.numpy.cosh(z))*dOmega_dz - Omega_z*(1/jax.numpy.cosh(z))*dU_dz)

      return summation[0]

# taking the 1st and 3rd derivative terms for the smoothness functions 
df1_dz = jax.grad(f1, argnums=3)
d3f1_dz3 = jax.grad(jax.grad(jax.grad(f1, argnums=3), argnums=3), argnums=3)

# vmapping
f1_vmap = jax.vmap(f1, in_axes=(None, None, None, 0, None))
df1_dz_vmap = jax.vmap(df1_dz, in_axes=(None, None, None, 0, None))
d3f1_dz3_vmap = jax.vmap(d3f1_dz3, in_axes=(None, None, None, 0, None))

# second equation residue

def f2(Omega_p, U_p, rng, z, lambda_val):
      dU_dz = dq_dz(U_p, rng, z, lambda_val)
      summation = (1/jax.numpy.cosh(z))*dU_dz-Hn(Omega_p, rng, z, lambda_val)
      return summation[0]

# taking the 1st and 3rd derivative terms for the smoothness functions 
df2_dz = jax.grad(f2, argnums=3)
d3f2_dz3 = jax.grad(jax.grad(jax.grad(f2, argnums=3), argnums=3), argnums=3)

# vmapping
f2_vmap = jax.vmap(f2, in_axes=(None, None, None, 0, None))
df2_dz_vmap = jax.vmap(df2_dz, in_axes=(None, None, None, 0, None))
d3f2_dz3_vmap = jax.vmap(d3f2_dz3, in_axes=(None, None, None, 0, None))

def equation_loss(Omega_p, U_p, rng, lambda_val):
      # first equation condition
      start_end = jnp.concatenate([jr.uniform(rng, 1, minval=-30, maxval=-20), jr.uniform(rng, 1, minval=20, maxval=30)])
      colloc_pts_1 = jnp.linspace(start_end[0], start_end[1], 80)
      colloc_pts_1 = colloc_pts_1.reshape(colloc_pts_1.shape[0],1) # reshaping for batch
      f1 = jnp.sum(f1_vmap(Omega_p, U_p, rng, colloc_pts_1, lambda_val)**2)/colloc_pts_1.shape[0]
      
      # second equation condition
      start_end = jnp.concatenate([jr.uniform(rng, 1, minval=-30, maxval=-29), jr.uniform(rng, 1, minval=29, maxval=30)])
      colloc_pts_2 = jnp.linspace(start_end[0], start_end[1], 80)
      colloc_pts_2 = colloc_pts_2.reshape(colloc_pts_2.shape[0],1)
      f2 = jnp.sum(f2_vmap(Omega_p, U_p, rng, colloc_pts_2, lambda_val)**2)/colloc_pts_2.shape[0]
      
      return (f1+f2)/2

def smoothness_loss(Omega_p, U_p, rng, lambda_val):
      colloc_pts = jr.uniform(rng, 80, minval=-1, maxval=1)
      colloc_pts = colloc_pts.reshape(colloc_pts.shape[0],1) # reshaping for batch
      # df2_dz to find the first smooth lambda value
      f1s = 1/colloc_pts.shape[0]*jnp.sum(jnp.abs(df1_dz_vmap(Omega_p, U_p, rng, colloc_pts, lambda_val))**2)
      f2s = 1/colloc_pts.shape[0]*jnp.sum(jnp.abs(df2_dz_vmap(Omega_p, U_p, rng, colloc_pts, lambda_val))**2)
      return (f1s+f2s)/2

# TODO - move colloc pt generation outside of loss functions?
def total_loss(trainable_state, rng):
      Omega_p = trainable_state['Omega_p']
      U_p = trainable_state['U_p']
      lambda_val = trainable_state['lambda']

      return (conditional_loss(Omega_p, rng, lambda_val) 
              + equation_loss(Omega_p, U_p, rng, lambda_val) 
              + smoothness_loss(Omega_p, U_p, rng, lambda_val))

CCF_val_and_grad = jax.jit(jax.value_and_grad(total_loss, argnums=0))

@partial(jax.jit, static_argnames=("optimizer",))
def adam_train_step(trainable_state, opt_state, rng, optimizer):
    # Get loss and gradients
    (loss_val, grads) = CCF_val_and_grad(trainable_state, rng)
    
    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state)
    trainable_state = optax.apply_updates(trainable_state, updates)
    
    return trainable_state, opt_state, loss_val

class InfoState(NamedTuple):
  iter_num: chex.Numeric

def lbfgs_loss(state):
  return total_loss(state, rng_lbfgs)

def print_info():
  def init_fn(params):
    del params
    return InfoState(iter_num=0)

  def update_fn(updates, state, params, *, value, grad, **extra_args):
    del params, extra_args

    jax.debug.print(
        'Iteration: {i}, Value: {v}, Gradient norm: {e}',
        i=state.iter_num,
        v=value,
        e=optax.tree.norm(grad),
    )
    return updates, InfoState(iter_num=state.iter_num + 1)

  return optax.GradientTransformationExtraArgs(init_fn, update_fn)

def run_opt(init_params, fun, opt, max_iter, tol):
  value_and_grad_fun = optax.value_and_grad_from_state(fun)

  def step(carry):
    params, state = carry
    value, grad = value_and_grad_fun(params, state=state)
    updates, state = opt.update(
        grad, state, params, value=value, grad=grad, value_fn=fun
    )
    params = optax.apply_updates(params, updates)
    return params, state

  def continuing_criterion(carry):
    _, state = carry
    iter_num = optax.tree.get(state, 'count')
    grad = optax.tree.get(state, 'grad')
    err = optax.tree.norm(grad)
    return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

  init_carry = (init_params, opt.init(init_params))
  final_params, final_state = jax.lax.while_loop(
      continuing_criterion, step, init_carry
  )
  return final_params, final_state


if __name__ == "__main__":
    # Training loop
    lambda_val = 1.3
    key = jax.random.key(11)
    key, key_init_Omega, key_init_U = jax.random.split(key, 3)
    Omega_p = q_DG_jax.init(key_init_Omega, jnp.array([[0.5]]), lambda_val)
    U_p = q_DG_jax.init(key_init_U, jnp.array([[0.5]]), lambda_val)

    trainable_state = {
        "Omega_p" : Omega_p,
        "U_p" : U_p,
        "lambda" : lambda_val
    }

    # First 100k iterations using ADAM
    LR = 0.001
    adam = optax.adam(LR)
    opt_state = adam.init(trainable_state)

    for i in range(10000):
        key, rng_step = jax.random.split(key)
        trainable_state, opt_state, loss_val = adam_train_step(trainable_state, opt_state, key, adam)

        if i % 1000 == 0:
            print(f"Step {i}, Loss: {loss_val}, Lambda: {trainable_state['lambda']}")

    rng_lbfgs = jax.random.key(0)

    # next 250k iterations using L-BFGS
    opt = optax.chain(print_info(), optax.lbfgs())
    final_params, _ = run_opt(trainable_state, lbfgs_loss, opt, max_iter=100, tol=1e-3)

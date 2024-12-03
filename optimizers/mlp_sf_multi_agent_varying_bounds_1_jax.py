import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import scipy

# Reproducibility
# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# torch.set_default_dtype(torch.float32)

import jax
from jax import jit
from functools import partial
import jax.numpy as jnp
from utils.bernstein_coeff_order10_arbitinterval import bernstein_coeff_order10_new


def nan_safe_div_value(x2, epsilon=1e-6):
	nudge = (x2 == 0) * epsilon
	return x2 + nudge


def nan_safe_pos_value(x2):
	return jnp.where(x2 < 0, 0, x2)


class mlp_sf_multi_agent_jax:
	
	def __init__(self, num_batch, num_agent):
		
		self.num_batch = num_batch
		self.num_agent = num_agent
		self.num_con = int(scipy.special.binom(self.num_agent,2))

		# P Matrices

		self.t_fin = 5
		self.num = 50

		self.t = self.t_fin/self.num
		
		tot_time = jnp.linspace(0, self.t_fin, self.num)
		self.tot_time = tot_time
		tot_time_copy = tot_time.reshape(self.num, 1)

		self.P, self.Pdot, self.Pddot = bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)
		self.P, self.Pdot, self.Pddot = jnp.asarray(self.P), jnp.asarray(self.Pdot), jnp.asarray(self.Pddot)

		# No. of Variables per agent
		self.nvar = self.P.shape[1]
		self.num = self.P.shape[0]
		self.num_batch = num_batch

		# self.A_via_points = self.P[49].reshape(1, self.nvar)
	
		self.a_agent = 0.4
		self.b_agent = 0.80
  

		self.v_max = 3.0
		self.v_min = 0.01

		self.a_max = 5.0
		
		# Parameters
		self.rho_agent = 1.1
		self.rho_ineq = 1.0
		self.rho_ell = 1.1
		t_fin = 5
		self.weight_smoothness = 30

		self.maxiter = 15
		self.tot_time = jnp.linspace(0, t_fin, self.num)

		# self.A_via_points = jnp.vstack([ self.P[29], self.P[59]  ])
		
		######## computing all the necessary matrices

		# Q = self.get_Q()
		# self.Q = Q

		self.A_projection  = jnp.eye(self.nvar*self.num_agent)

		
		A_fc = self.get_A_fc()
		self.A_fc = A_fc
		
		A_eq = self.get_A_eq()
		self.A_eq = A_eq

		A_eq_init = self.get_A_eq_init()
		self.A_eq_init = A_eq_init
  
		A_ell = self.get_A_ell()
		self.A_ell = A_ell

		A_v = self.get_A_v()
		self.A_v = A_v
  
		A_a  = self.get_A_a()
		self.A_a = A_a

		Q_tilda_inv_init = self.get_Q_tilde_inv_init(A_eq_init)
		self.Q_tilda_inv_init = Q_tilda_inv_init


		Q_tilda_inv = self.get_Q_tilde_inv(A_fc, A_eq)
		self.Q_tilda_inv = Q_tilda_inv

		self.A_acc_mat = self.get_A_acc_mat()
		
		self.compute_boundary_vec_batch = jax.vmap(self.compute_boundary_vec_single, in_axes = (0, 0, 0 )  )

		# self.num_via_xyz = 5
		self.num_via_xyz = 2
		self.num_via_points = 3 * self.num_via_xyz
		self.epsilon = 0.000001
		self.num_eq = self.A_eq.shape[0]
  
  
	def get_A_ell(self):

		return jnp.kron(jnp.eye(self.num_agent), self.P )

	def get_A_v(self):

		return jnp.kron(jnp.eye(self.num_agent), self.Pdot )

	def get_A_a(self):

		return jnp.kron(jnp.eye(self.num_agent), self.Pddot )

	def get_A_acc_mat(self):

		A_acc_mat = jnp.kron(jnp.eye(self.num_agent), self.Pddot) ################### smoothness cost of all agents stacked up

		return A_acc_mat

	def get_A_fc(self):

		def get_A_r(i):
			tmp = np.hstack((np.zeros((i, self.num_agent-i)), -np.eye(i)) )
			tmp[:,self.num_agent-i-1]=1
			return tmp

		tmp = np.vstack([get_A_r(i) for i in jnp.arange(self.num_agent-1, 0, -1)])

		return np.kron(tmp, self.P)

	def get_A_eq(self):

		return jnp.kron(jnp.eye(self.num_agent), jnp.vstack([self.P[0], self.Pdot[0], self.Pddot[0], self.P[-1], self.Pdot[-1], self.Pddot[-1] ]))


	def get_Q_tilde_inv(self, A_fc, A_eq):
	#  +self.rho_ineq*jnp.mm(self.A_a.T, self.A_a)
	 
		Q = jnp.linalg.inv(jnp.vstack((jnp.hstack(( jnp.dot(self.A_projection.T, self.A_projection)  + self.rho_agent*jnp.dot(A_fc.T, A_fc)+self.rho_ell*jnp.dot(self.A_ell.T, self.A_ell), A_eq.T)), 
									 jnp.hstack((A_eq, jnp.zeros((A_eq.shape[0], A_eq.shape[0])  ) )  ))))	
  
		# Q = jnp.linalg.inv(jnp.vstack((jnp.hstack(( jnp.mm(self.A_projection.T, self.A_projection)  + self.rho_agent*jnp.mm(A_fc.T, A_fc)+self.rho_ell*jnp.dot(self.A_ell.T, self.A_ell), A_eq.T)), 
		# 							 jnp.hstack((A_eq, jnp.zeros((A_eq.size(dim = 0), A_eq.size(dim = 0))  ) )  ))))	
		
		return Q

	def get_Q_tilde_inv_init(self, A_eq_init):
		
		return jnp.linalg.inv(jnp.vstack((jnp.hstack(( jnp.dot(self.A_projection.T, self.A_projection) , A_eq_init.T)), 
									 jnp.hstack((A_eq_init, jnp.zeros((A_eq_init.shape[0], A_eq_init.shape[0])  ) )  ))))	
 
	def get_A_eq_init(self):

		return jnp.kron(jnp.eye(self.num_agent), jnp.vstack([self.P[0], self.Pdot[0], self.Pddot[0], self.P[-1], self.Pdot[-1], self.Pddot[-1] ] ))

	@partial(jit, static_argnums=(0,))
	@partial(jax.vmap, in_axes=(None,)+tuple(0 for _ in range(6)))
	def qp_layer_init(self, b_eq_x_init, b_eq_y_init, b_eq_z_init, c_x_pred, c_y_pred, c_z_pred):

		lincost_x = jnp.zeros((self.nvar*self.num_agent  ))-jnp.dot(self.A_projection.T, c_x_pred.T).T
		lincost_y = jnp.zeros((self.nvar*self.num_agent  ))-jnp.dot(self.A_projection.T, c_y_pred.T).T
		lincost_z = jnp.zeros((self.nvar*self.num_agent  ))-jnp.dot(self.A_projection.T, c_z_pred.T).T
		
					
		sol_x = jnp.dot(self.Q_tilda_inv_init, jnp.hstack([-lincost_x, b_eq_x_init]).T).T
		sol_y = jnp.dot(self.Q_tilda_inv_init, jnp.hstack([-lincost_y, b_eq_y_init]).T).T
		sol_z = jnp.dot(self.Q_tilda_inv_init, jnp.hstack([-lincost_z, b_eq_z_init]).T).T
		
		primal_sol_x = sol_x[0:self.nvar*self.num_agent]
		primal_sol_y = sol_y[0:self.nvar*self.num_agent]
		primal_sol_z = sol_z[0:self.nvar*self.num_agent]

		return primal_sol_x, primal_sol_y, primal_sol_z 

	def compute_boundary_vec_single(self, state_x, state_y, state_z):

		b_eq_x = state_x.reshape(6, self.num_agent).T
		b_eq_y = state_y.reshape(6, self.num_agent).T 
		b_eq_z = state_z.reshape(6, self.num_agent).T 
		

		b_eq_x = b_eq_x.reshape(self.num_agent*(6))
		b_eq_y = b_eq_y.reshape(self.num_agent*(6))
		b_eq_z = b_eq_z.reshape(self.num_agent*(6))

		return b_eq_x, b_eq_y, b_eq_z
		
	# @partial(jit, static_argnums=(0,))
	# @partial(jax.vmap, in_axes=(None,)+tuple(0 for _ in range(21)))
	def compute_x(self, c_x_pred, c_y_pred, c_z_pred, d_agent, alpha_agent, beta_agent, lamda_x, lamda_y, lamda_z, b_eq_x, b_eq_y, b_eq_z,  x_ell, y_ell, z_ell, a_ell, b_ell, c_ell, alpha_ell, beta_ell, d_ell):

		b_agent_x = self.a_agent*d_agent*jnp.cos(alpha_agent)*jnp.sin(beta_agent)
		b_agent_y = self.a_agent*d_agent*jnp.sin(alpha_agent)*jnp.sin(beta_agent)
		b_agent_z = self.b_agent*d_agent*jnp.cos(beta_agent)
  
		b_ell_x = x_ell + a_ell*d_ell*jnp.cos(alpha_ell)*jnp.sin(beta_ell)
		b_ell_y = y_ell + b_ell*d_ell*jnp.sin(alpha_ell)*jnp.sin(beta_ell)
		b_ell_z = z_ell + c_ell*d_ell*jnp.cos(beta_ell)
		
		lincost_x = -lamda_x-jnp.dot(self.A_projection.T, c_x_pred.T).T-self.rho_agent*jnp.dot(self.A_fc.T, b_agent_x.T).T-self.rho_ell*jnp.dot(self.A_ell.T, b_ell_x.T).T
		lincost_y = -lamda_y-jnp.dot(self.A_projection.T, c_y_pred.T).T-self.rho_agent*jnp.dot(self.A_fc.T, b_agent_y.T).T-self.rho_ell*jnp.dot(self.A_ell.T, b_ell_y.T).T
		lincost_z = -lamda_z-jnp.dot(self.A_projection.T, c_z_pred.T).T-self.rho_agent*jnp.dot(self.A_fc.T, b_agent_z.T).T-self.rho_ell*jnp.dot(self.A_ell.T, b_ell_z.T).T
  
		sol_x = jnp.dot(self.Q_tilda_inv, jnp.hstack((-lincost_x, b_eq_x)).T).T
		sol_y = jnp.dot(self.Q_tilda_inv, jnp.hstack((-lincost_y, b_eq_y)).T).T
		sol_z = jnp.dot(self.Q_tilda_inv, jnp.hstack((-lincost_z, b_eq_z)).T).T

		################## from here
		primal_sol_x = sol_x[0:self.num_agent*self.nvar]
		primal_sol_y = sol_y[0:self.num_agent*self.nvar]
		primal_sol_z = sol_z[0:self.num_agent*self.nvar]
		
		return primal_sol_x, primal_sol_y, primal_sol_z 
		
	# @partial(jit, static_argnums=(0,))
	# @partial(jax.vmap, in_axes=(None,)+tuple(0 for _ in range(12)))
	def compute_alpha_d(self, primal_sol_x, primal_sol_y, primal_sol_z, lamda_x, lamda_y, lamda_z, x_ell, y_ell, z_ell, a_ell, b_ell, c_ell):
	 
		##############computing xi-xj

		wc_alpha_agent = jnp.dot(self.A_fc, primal_sol_x.T).T ### xi-xj
  
		ws_alpha_agent = jnp.dot(self.A_fc, primal_sol_y.T).T ### yi-yj
		alpha_agent = jnp.arctan2(ws_alpha_agent, nan_safe_div_value(wc_alpha_agent, epsilon=self.epsilon))
  
		wc_beta_agent = jnp.dot(self.A_fc, primal_sol_z.T).T ### zi-zj

		r_agent = jnp.sqrt((wc_alpha_agent**2/self.a_agent**2)+(ws_alpha_agent**2/self.a_agent**2)+(wc_beta_agent**2/self.b_agent**2) +self.epsilon)
		beta_agent = jnp.arccos(wc_beta_agent/(self.b_agent*r_agent+self.epsilon))

		# c_1_d_agent = 1.0*self.a_agent**2*jnp.sin(beta_agent)**2 - 1.0*self.b_agent**2*jnp.sin(beta_agent)**2 + 1.0*self.b_agent**2
		# c_2_d_agent = 1.0*self.a_agent*wc_alpha_agent*jnp.sin(beta_agent)*jnp.cos(alpha_agent) + 1.0*self.a_agent*ws_alpha_agent*jnp.sin(alpha_agent)*jnp.sin(beta_agent) + 1.0*self.b_agent*wc_beta_agent*jnp.cos(beta_agent)

		# d_agent_temp = c_2_d_agent/c_1_d_agent
  
		d_agent_temp = r_agent
  
		d_agent = jnp.maximum(jnp.ones((self.num_con*(self.num))), d_agent_temp )
  
	
		res_x_agent = wc_alpha_agent-(self.a_agent)*d_agent*jnp.cos(alpha_agent)*jnp.sin(beta_agent)
		res_y_agent = ws_alpha_agent-(self.a_agent)*d_agent*jnp.sin(alpha_agent)*jnp.sin(beta_agent)
		res_z_agent = wc_beta_agent-(self.b_agent)*d_agent*jnp.cos(beta_agent)

		######################## Elliptical bounds
  
  
		wc_alpha_ell = jnp.dot(self.A_ell, primal_sol_x.T).T-x_ell # *jnp.ones((self.num_batch, self.num_agent*self.num)) 
		ws_alpha_ell = jnp.dot(self.A_ell, primal_sol_y.T).T-y_ell # *jnp.ones((self.num_batch, self.num_agent*self.num)) 
		alpha_ell = jnp.arctan2(a_ell*ws_alpha_ell, nan_safe_div_value(b_ell*wc_alpha_ell, epsilon=self.epsilon))
  
		wc_beta_ell = jnp.dot(self.A_ell, primal_sol_z.T).T-z_ell #*jnp.ones((self.num_batch, self.num_agent*self.num)) 
  
		r_ell = jnp.sqrt((wc_alpha_ell**2/a_ell**2)+(ws_alpha_ell**2/b_ell**2)+(wc_beta_ell**2/c_ell**2)+self.epsilon )
		beta_ell = jnp.arccos(wc_beta_ell/(c_ell*r_ell+self.epsilon))
  
  
		# c1_ell =  1.0*a_ell**2*self.rho_ell*jnp.sin(beta_ell)**2*jnp.cos(alpha_ell)**2 + 2*b_ell**2*jnp.sin(alpha_ell)**2*jnp.sin(beta_ell)**2 + 2*c_ell**2*jnp.cos(beta_ell)**2	
		# c2_ell =  1.0*a_ell*self.rho_ell*wc_alpha_ell*jnp.sin(beta_ell)*jnp.cos(alpha_ell) + 2.0*b_ell*ws_alpha_ell*jnp.sin(alpha_ell)*jnp.sin(beta_ell) + 2.0*c_ell*wc_beta_ell*jnp.cos(beta_ell)

		# d_ell = c2_ell/c1_ell
  
		d_ell = r_ell
		d_ell = jnp.minimum(jnp.ones((self.num*(self.num_agent))), d_ell)
  
		res_x_ell = wc_alpha_ell-(a_ell)*d_ell*jnp.cos(alpha_ell)*jnp.sin(beta_ell)
		res_y_ell = ws_alpha_ell-(b_ell)*d_ell*jnp.sin(alpha_ell)*jnp.sin(beta_ell)
		res_z_ell = wc_beta_ell-(c_ell)*d_ell*jnp.cos(beta_ell)

	
		lamda_x = lamda_x-self.rho_agent*jnp.dot(self.A_fc.T, res_x_agent.T).T-self.rho_ell*jnp.dot(self.A_ell.T, res_x_ell.T).T
		lamda_y = lamda_y-self.rho_agent*jnp.dot(self.A_fc.T, res_y_agent.T).T-self.rho_ell*jnp.dot(self.A_ell.T, res_y_ell.T).T
		lamda_z = lamda_z-self.rho_agent*jnp.dot(self.A_fc.T, res_z_agent.T).T-self.rho_ell*jnp.dot(self.A_ell.T, res_z_ell.T).T
  
		primal_residual = (
			jnp.linalg.norm(res_x_agent) + jnp.linalg.norm(res_y_agent) + jnp.linalg.norm(res_z_agent) + 
			jnp.linalg.norm(res_x_ell) + jnp.linalg.norm(res_y_ell) + jnp.linalg.norm(res_z_ell)
		)

		return alpha_agent, beta_agent, d_agent, lamda_x, lamda_y, lamda_z, primal_residual, alpha_ell, beta_ell, d_ell
	
	@partial(jit, static_argnums=(0,))
	@partial(jax.vmap, in_axes=(None,)+tuple(0 for _ in range(18)))	
	def custom_forward(self, primal_sol_x, primal_sol_y, primal_sol_z, lamda_x, lamda_y, lamda_z, state_x, state_y, state_z, c_x_pred, c_y_pred, c_z_pred, x_ell, y_ell, z_ell, a_ell, b_ell, c_ell):
		

		b_eq_x, b_eq_y, b_eq_z = self.compute_boundary_vec_single(state_x, state_y, state_z)

		######### one iteration before the main loop

		alpha_agent, beta_agent, d_agent, lamda_x, lamda_y, lamda_z, primal_residual, alpha_ell, beta_ell, d_ell = self.compute_alpha_d(primal_sol_x, primal_sol_y, primal_sol_z, lamda_x, lamda_y, lamda_z, x_ell, y_ell, z_ell, a_ell, b_ell, c_ell)
		primal_sol_x, primal_sol_y, primal_sol_z = self.compute_x(c_x_pred, c_y_pred, c_z_pred, d_agent, alpha_agent, beta_agent, lamda_x, lamda_y, lamda_z, b_eq_x, b_eq_y, b_eq_z,  x_ell, y_ell, z_ell, a_ell, b_ell, c_ell, alpha_ell, beta_ell, d_ell)

		################################################################################################

		alpha_agent_init = alpha_agent 
		beta_agent_init = beta_agent 
		d_agent_init = d_agent
		lamda_x_init = lamda_x  
		lamda_y_init = lamda_y 
		lamda_z_init = lamda_z  

		alpha_ell_init = alpha_ell
		beta_ell_init = beta_ell 
		d_ell_init = d_ell
  
		primal_sol_x_init = primal_sol_x
		primal_sol_y_init = primal_sol_y
		primal_sol_z_init = primal_sol_z


		def lax_custom_forward(carry, idx):
	  
			primal_sol_x, primal_sol_y, primal_sol_z, alpha_agent, beta_agent, d_agent, lamda_x, lamda_y, lamda_z, alpha_ell, beta_ell, d_ell = carry	
	  
			primal_sol_x_prev = primal_sol_x
			primal_sol_y_prev = primal_sol_y
			primal_sol_z_prev = primal_sol_z
   
			lamda_x_prev = lamda_x
			lamda_y_prev = lamda_y
			lamda_z_prev = lamda_z
   
			alpha_agent, beta_agent, d_agent, lamda_x, lamda_y, lamda_z, primal_residual, alpha_ell, beta_ell, d_ell = self.compute_alpha_d(primal_sol_x, primal_sol_y, primal_sol_z, lamda_x, lamda_y, lamda_z, x_ell, y_ell, z_ell, a_ell, b_ell, c_ell)
			primal_sol_x, primal_sol_y, primal_sol_z = self.compute_x(c_x_pred, c_y_pred, c_z_pred, d_agent, alpha_agent, beta_agent, lamda_x, lamda_y, lamda_z, b_eq_x, b_eq_y, b_eq_z,  x_ell, y_ell, z_ell, a_ell, b_ell, c_ell, alpha_ell, beta_ell, d_ell)

			dual_residual = (
				jnp.linalg.norm(primal_sol_x_prev - primal_sol_x) +
				jnp.linalg.norm(primal_sol_y_prev - primal_sol_y) +
				jnp.linalg.norm(primal_sol_z_prev - primal_sol_z) +
				jnp.linalg.norm(lamda_x_prev - lamda_x) +
				jnp.linalg.norm(lamda_y_prev - lamda_y) +
				jnp.linalg.norm(lamda_z_prev - lamda_z)
			)

			return (primal_sol_x, primal_sol_y, primal_sol_z, alpha_agent, beta_agent, d_agent, lamda_x, lamda_y, lamda_z, alpha_ell, beta_ell, d_ell), (primal_residual, dual_residual)

		carry_init = (primal_sol_x_init, primal_sol_y_init, primal_sol_z_init, alpha_agent_init, beta_agent_init, d_agent_init, lamda_x_init, lamda_y_init, lamda_z_init, alpha_ell_init, beta_ell_init, d_ell_init)
		
		carry_final, res_tot = jax.lax.scan(lax_custom_forward, carry_init, jnp.arange(self.maxiter))

		primal_sol_x, primal_sol_y, primal_sol_z, alpha_agent, beta_agent, d_agent, lamda_x, lamda_y, lamda_z, alpha_ell, beta_ell, d_ell = carry_final 
		res_primal_stack, res_fixed_point_stack = res_tot

		accumulated_res_primal = jnp.sum(res_primal_stack, axis=0)/self.maxiter
		accumulated_res_fixed_point = jnp.sum(res_fixed_point_stack, axis=0)/self.maxiter	
  
		return primal_sol_x, primal_sol_y, primal_sol_z, accumulated_res_primal, accumulated_res_fixed_point, res_primal_stack, res_fixed_point_stack


	@partial(jit, static_argnums=(0,))	
	def __call__(self, c_x_pred, c_y_pred, c_z_pred, c_x_guess, c_y_guess, c_z_guess, lamda_x_init, lamda_y_init, lamda_z_init, state_x, state_y, state_z, x_ell, y_ell, z_ell, a_ell, b_ell, c_ell):
  
		b_eq_x_init, b_eq_y_init, b_eq_z_init = self.compute_boundary_vec_batch(state_x, state_y, state_z)

		primal_sol_x_init, primal_sol_y_init, primal_sol_z_init = self.qp_layer_init(
			b_eq_x_init, b_eq_y_init, b_eq_z_init, c_x_guess, c_y_guess, c_z_guess
		)
		
		primal_sol_x, primal_sol_y, primal_sol_z, accumulated_res_primal, accumulated_res_fixed_point, res_primal_stack, res_fixed_point_stack = self.custom_forward(
			primal_sol_x_init, primal_sol_y_init, primal_sol_z_init, 
			lamda_x_init, lamda_y_init, lamda_z_init, 
			state_x, state_y, state_z, c_x_pred, c_y_pred, c_z_pred, 
			x_ell, y_ell, z_ell, a_ell, b_ell, c_ell
		)

		return primal_sol_x, primal_sol_y, primal_sol_z, accumulated_res_primal, accumulated_res_fixed_point, res_primal_stack.T, res_fixed_point_stack.T


	
	
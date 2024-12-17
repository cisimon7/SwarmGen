import jax.numpy as jnp
import numpy as np
import scipy
from jax import jit, random, vmap
from utils.bernstein_coeff_order10_arbitinterval import bernstein_coeff_order10_new
from functools import partial
import scipy
import matplotlib.pyplot as plt
import time

from scipy.io import loadmat
import jax.lax as lax
import jax



class multi_agent_qp_node_base():

	def __init__(self, num_batch, num_agent):

		self.num_agent = num_agent
		self.num_batch = num_batch
		self.num_con = int(scipy.special.binom(self.num_agent,2))
		self.a_agent = 0.40 # crazy fly dimensions
		self.b_agent = 0.80

		self.t_fin = 5
		self.num = 50

		self.t = self.t_fin/self.num
		
		tot_time = np.linspace(0, self.t_fin, self.num)
		self.tot_time = tot_time
		tot_time_copy = tot_time.reshape(self.num, 1)
		
		self.P, self.Pdot, self.Pddot = bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)

		self.P_jax, self.Pdot_jax, self.Pddot_jax = jnp.asarray(self.P), jnp.asarray(self.Pdot), jnp.asarray(self.Pddot)

		self.nvar = jnp.shape(self.P_jax)[1]

		###################### parameters

		self.rho_agent = 2.0
		self.rho_ineq = 1.0
		self.rho_bounds = 1
		self.rho_ell = 2.0
		self.weight_smoothness = 5.0
		
		# self.weight_vdes = 1.0
		# self.weight_ydes = 30.0 

		self.v_max = 3.0
		self.v_min = 0.01

		self.a_max = 5.0
  
		# self.wheel_base = 2.5
		# self.num_mean_update = 4
  
		#################################################
		self.maxiter = 1000

		######## computing all the necessary matrices
  

		
		Q_x = self.get_Q_x()
		self.Q_x = jnp.asarray(Q_x)
  
		Q_y = self.get_Q_y()
		self.Q_y = jnp.asarray(Q_y)
  
		Q_z = self.get_Q_z()
		self.Q_z = jnp.asarray(Q_z)
  	 	
		A_fc = self.get_A_fc()
		self.A_fc = jnp.asarray(A_fc)

		
		A_v = self.get_A_v()
		self.A_v = jnp.asarray(A_v)
  
		A_a = self.get_A_a()
		self.A_a = jnp.asarray(A_a)
		
		A_eq_x = self.get_A_eq_x()
		self.A_eq_x = jnp.asarray(A_eq_x)
  
		A_eq_y = self.get_A_eq_y()
		self.A_eq_y = jnp.asarray(A_eq_y)
  
		A_eq_z = self.get_A_eq_y()
		self.A_eq_z = jnp.asarray(A_eq_z)
  
		A_ell =  self.get_A_ell()
		self.A_ell = A_ell

		self.Q_tilda_inv_x, self.Q_tilda_inv_y, self.Q_tilda_inv_z = self.get_Q_tilde_inv(Q_x, Q_y, Q_z, A_fc, A_eq_x, A_eq_y, A_eq_z)
		self.compute_boundary_vec_batch = (vmap(self.compute_boundary_vec_single, in_axes = (0, 0, 0)  ))

	

 
	def get_A_ell(self):

		return np.kron(np.identity(self.num_agent), self.P )
	
 	
	def get_A_v(self):

		return np.kron(np.identity(self.num_agent), self.Pdot )

	def get_A_a(self):

		return np.kron(np.identity(self.num_agent), self.Pddot )
	
  
	def get_Q_x(self):

		Q = np.kron(np.identity(self.num_agent), self.weight_smoothness*np.dot(self.Pddot.T, self.Pddot)      ) ################### smoothness cost and desired velocity cost of all agents stacked up

		return Q


	def get_Q_y(self):

		Q = np.kron(np.identity(self.num_agent), self.weight_smoothness*np.dot(self.Pddot.T, self.Pddot)      ) ################### smoothness cost and lane merging cost of all agents stacked up

		return Q


	def get_Q_z(self):

		Q = np.kron(np.identity(self.num_agent), 5*self.weight_smoothness*np.dot(self.Pddot.T, self.Pddot)      ) ################### smoothness cost and lane merging cost of all agents stacked up

		return Q

	
	def get_A_fc(self):

		def get_A_r(i):
			tmp = np.hstack((np.zeros((i, self.num_agent-i)), -np.identity(i)))
			tmp[:,self.num_agent-i-1]=1
			return tmp

		tmp = np.vstack([get_A_r(i) for i in np.arange(self.num_agent-1, 0, -1)])

		return np.kron(tmp, self.P)


	def get_A_eq_x(self):

		return np.kron(np.identity(self.num_agent), np.vstack((self.P[0], self.Pdot[0], self.Pddot[0], self.P[-1], self.Pdot[-1], self.Pddot[-1]  )))


	def get_A_eq_y(self):

		return np.kron(np.identity(self.num_agent), np.vstack((self.P[0], self.Pdot[0], self.Pddot[0], self.P[-1], self.Pdot[-1], self.Pddot[-1]  )))


	def get_A_eq_z(self):

		return np.kron(np.identity(self.num_agent), np.vstack((self.P[0], self.Pdot[0], self.Pddot[0], self.P[-1], self.Pdot[-1], self.Pddot[-1]  )))

	

	def get_Q_tilde_inv(self, Q_x, Q_y, Q_z, A_fc, A_eq_x, A_eq_y, A_eq_z):
	 
		# Q_tilda_inv_x = np.linalg.inv(np.vstack((np.hstack(( Q_x + self.rho_agent*np.dot(A_fc.T, A_fc)+self.rho_ineq*np.dot(self.A_v.T, self.A_v)+self.rho_ineq*np.dot(self.A_a.T, self.A_a)+self.rho_ell*jnp.dot(self.A_ell.T, self.A_ell), A_eq_x.T)  ), 
		# 							 np.hstack((A_eq_x, np.zeros((np.shape(A_eq_x)[0], np.shape(A_eq_x)[0])))))))	
  
		# Q_tilda_inv_y = np.linalg.inv(np.vstack((np.hstack(( Q_y + self.rho_agent*np.dot(A_fc.T, A_fc)+self.rho_ineq*np.dot(self.A_v.T, self.A_v)+self.rho_ineq*np.dot(self.A_a.T, self.A_a)+self.rho_ell*jnp.dot(self.A_ell.T, self.A_ell), A_eq_y.T)), 
		# 							 np.hstack((A_eq_y, np.zeros((np.shape(A_eq_y)[0], np.shape(A_eq_y)[0])))))))	
  
		# Q_tilda_inv_z = np.linalg.inv(np.vstack((np.hstack(( Q_z + self.rho_agent*np.dot(A_fc.T, A_fc)+self.rho_ineq*np.dot(self.A_v.T, self.A_v)+self.rho_ineq*np.dot(self.A_a.T, self.A_a)+self.rho_ell*jnp.dot(self.A_ell.T, self.A_ell), A_eq_z.T)), 
		# 							 np.hstack((A_eq_z, np.zeros((np.shape(A_eq_z)[0], np.shape(A_eq_z)[0])))))))
  
		Q_tilda_inv_x = np.linalg.inv(np.vstack((np.hstack(( Q_x + self.rho_agent*np.dot(A_fc.T, A_fc)+self.rho_ell*jnp.dot(self.A_ell.T, self.A_ell), A_eq_x.T)  ), 
									 np.hstack((A_eq_x, np.zeros((np.shape(A_eq_x)[0], np.shape(A_eq_x)[0])))))))	
  
		Q_tilda_inv_y = np.linalg.inv(np.vstack((np.hstack(( Q_y + self.rho_agent*np.dot(A_fc.T, A_fc)+self.rho_ell*jnp.dot(self.A_ell.T, self.A_ell), A_eq_y.T)), 
									 np.hstack((A_eq_y, np.zeros((np.shape(A_eq_y)[0], np.shape(A_eq_y)[0])))))))	
  
		Q_tilda_inv_z = np.linalg.inv(np.vstack((np.hstack(( Q_z + self.rho_agent*np.dot(A_fc.T, A_fc)+self.rho_ell*jnp.dot(self.A_ell.T, self.A_ell), A_eq_z.T)), 
									 np.hstack((A_eq_z, np.zeros((np.shape(A_eq_z)[0], np.shape(A_eq_z)[0])))))))
  	
		return Q_tilda_inv_x, Q_tilda_inv_y, Q_tilda_inv_z


	@partial(jit, static_argnums=(0,))
	def compute_boundary_vec_single(self, state_x, state_y, state_z ):
	
		b_eq_x = state_x.reshape(6, self.num_agent).T 
		b_eq_y = state_y.reshape(6, self.num_agent).T 
		b_eq_z = state_z.reshape(6, self.num_agent).T 
  

		b_eq_x = b_eq_x.reshape(self.num_agent*6)
		b_eq_y = b_eq_y.reshape(self.num_agent*6)
		b_eq_z = b_eq_z.reshape(self.num_agent*6)

		return b_eq_x, b_eq_y, b_eq_z


	@partial(jit, static_argnums=(0,))
	def compute_bounds(self, x_min, x_max, y_min, y_max, z_min, z_max):
		
		
		x_min_vec = jnp.ones(self.num*self.num_agent)*x_min
		x_max_vec = jnp.ones(self.num*self.num_agent)*x_max

		y_min_vec = jnp.ones(self.num*self.num_agent)*y_min
		y_max_vec = jnp.ones(self.num*self.num_agent)*y_max

		z_min_vec = jnp.ones(self.num*self.num_agent)*z_min
		z_max_vec = jnp.ones(self.num*self.num_agent)*z_max
		
 
		
		return x_min_vec, x_max_vec, y_min_vec, y_max_vec, z_min_vec, z_max_vec



	@partial(jit, static_argnums=(0,))	
	def compute_x(self, d_agent, alpha_agent, beta_agent, lamda_x, lamda_y, lamda_z, b_eq_x, b_eq_y, b_eq_z, alpha_v, beta_v, d_v, alpha_a, beta_a, d_a, x_ell, y_ell, z_ell, a_ell, b_ell, c_ell, alpha_ell, beta_ell, d_ell):

		b_agent_x = self.a_agent*d_agent*jnp.cos(alpha_agent)*jnp.sin(beta_agent)
		b_agent_y = self.a_agent*d_agent*jnp.sin(alpha_agent)*jnp.sin(beta_agent)
		b_agent_z = self.b_agent*d_agent*jnp.cos(beta_agent)
  
		# b_ell_x = x_ell*jnp.ones(( self.num_batch, self.num*self.num_agent )) + a_ell*d_ell*jnp.cos(alpha_ell)*jnp.sin(beta_ell)
		# b_ell_y = y_ell*jnp.ones(( self.num_batch, self.num*self.num_agent )) + b_ell*d_ell*jnp.sin(alpha_ell)*jnp.sin(beta_ell)
		# b_ell_z = z_ell*jnp.ones(( self.num_batch, self.num*self.num_agent )) + c_ell*d_ell*jnp.cos(beta_ell)
  
		b_ell_x = x_ell + a_ell*d_ell*jnp.cos(alpha_ell)*jnp.sin(beta_ell)
		b_ell_y = y_ell + b_ell*d_ell*jnp.sin(alpha_ell)*jnp.sin(beta_ell)
		b_ell_z = z_ell + c_ell*d_ell*jnp.cos(beta_ell)
  

		b_ax = d_a*jnp.cos(alpha_a)*jnp.sin(beta_a)
		b_ay = d_a*jnp.sin(alpha_a)*jnp.sin(beta_a)
		b_az = d_a*jnp.cos(beta_a)

		b_vx = d_v*jnp.cos(alpha_v)*jnp.sin(beta_v)
		b_vy = d_v*jnp.sin(alpha_v)*jnp.sin(beta_v)
		b_vz = d_v*jnp.cos(beta_v)
	
 	
		# lincost_x = -lamda_x-self.rho_agent*jnp.dot(self.A_fc.T, b_agent_x.T).T-self.rho_ineq*jnp.dot(self.A_v.T, b_vx.T).T-self.rho_ineq*jnp.dot(self.A_a.T, b_ax.T).T-self.rho_ell*jnp.dot(self.A_ell.T, b_ell_x.T).T
		# lincost_y = -lamda_y-self.rho_agent*jnp.dot(self.A_fc.T, b_agent_y.T).T-self.rho_ineq*jnp.dot(self.A_v.T, b_vy.T).T-self.rho_ineq*jnp.dot(self.A_a.T, b_ay.T).T-self.rho_ell*jnp.dot(self.A_ell.T, b_ell_y.T).T
		# lincost_z = -lamda_z-self.rho_agent*jnp.dot(self.A_fc.T, b_agent_z.T).T-self.rho_ineq*jnp.dot(self.A_v.T, b_vz.T).T-self.rho_ineq*jnp.dot(self.A_a.T, b_az.T).T-self.rho_ell*jnp.dot(self.A_ell.T, b_ell_z.T).T
  	
		lincost_x = -lamda_x-self.rho_agent*jnp.dot(self.A_fc.T, b_agent_x.T).T-self.rho_ell*jnp.dot(self.A_ell.T, b_ell_x.T).T
		lincost_y = -lamda_y-self.rho_agent*jnp.dot(self.A_fc.T, b_agent_y.T).T-self.rho_ell*jnp.dot(self.A_ell.T, b_ell_y.T).T
		lincost_z = -lamda_z-self.rho_agent*jnp.dot(self.A_fc.T, b_agent_z.T).T-self.rho_ell*jnp.dot(self.A_ell.T, b_ell_z.T).T
  			
  
		sol_x = jnp.dot(self.Q_tilda_inv_x, jnp.hstack(( -lincost_x, b_eq_x )).T).T
		sol_y = jnp.dot(self.Q_tilda_inv_y, jnp.hstack(( -lincost_y, b_eq_y )).T).T
		sol_z = jnp.dot(self.Q_tilda_inv_z, jnp.hstack(( -lincost_z, b_eq_z )).T).T
  

		################## from here
		primal_sol_x = sol_x[:, 0:self.num_agent*self.nvar]
		primal_sol_y = sol_y[:, 0:self.num_agent*self.nvar]
		primal_sol_z = sol_z[:, 0:self.num_agent*self.nvar]
		
		return primal_sol_x, primal_sol_y, primal_sol_z 

	
	@partial(jit, static_argnums=(0,))		
	def compute_alpha_d(self, primal_sol_x, primal_sol_y, primal_sol_z, lamda_x, lamda_y, lamda_z, x_ell, y_ell, z_ell, a_ell, b_ell, c_ell):
	 
		##############computing xi-xj

		wc_alpha_agent = jnp.dot(self.A_fc, primal_sol_x.T).T ### xi-xj
  
		ws_alpha_agent = jnp.dot(self.A_fc, primal_sol_y.T).T ### yi-yj
		alpha_agent = jnp.arctan2(ws_alpha_agent, wc_alpha_agent )
  
		wc_beta_agent = jnp.dot(self.A_fc, primal_sol_z.T).T ### zi-zj

		r_agent = jnp.sqrt((wc_alpha_agent**2/self.a_agent**2)+(ws_alpha_agent**2/self.a_agent**2)+(wc_beta_agent**2/self.b_agent**2) )
		beta_agent = jnp.arccos(wc_beta_agent/(self.b_agent*r_agent))

		# c_1_d_agent = 1.0*self.a_agent**2*jnp.sin(beta_agent)**2 - 1.0*self.b_agent**2*jnp.sin(beta_agent)**2 + 1.0*self.b_agent**2
		# c_2_d_agent = 1.0*self.a_agent*wc_alpha_agent*jnp.sin(beta_agent)*jnp.cos(alpha_agent) + 1.0*self.a_agent*ws_alpha_agent*jnp.sin(alpha_agent)*jnp.sin(beta_agent) + 1.0*self.b_agent*wc_beta_agent*jnp.cos(beta_agent)

		# d_agent_temp = c_2_d_agent/c_1_d_agent
  
		d_agent_temp = r_agent
  
		d_agent = jnp.maximum(jnp.ones((self.num_batch, self.num_con*(self.num) )), d_agent_temp )
  
	
		res_x_agent = wc_alpha_agent-(self.a_agent)*d_agent*jnp.cos(alpha_agent)*jnp.sin(beta_agent)
		res_y_agent = ws_alpha_agent-(self.a_agent)*d_agent*jnp.sin(alpha_agent)*jnp.sin(beta_agent)
		res_z_agent = wc_beta_agent-(self.b_agent)*d_agent*jnp.cos(beta_agent)
	
		######################## Elliptical bounds
  
  
		wc_alpha_ell = jnp.dot(self.A_ell, primal_sol_x.T).T-x_ell#*jnp.ones((self.num_batch, self.num_agent*self.num)) 
		ws_alpha_ell = jnp.dot(self.A_ell, primal_sol_y.T).T-y_ell#*jnp.ones((self.num_batch, self.num_agent*self.num)) 
		alpha_ell = jnp.arctan2(a_ell*ws_alpha_ell, b_ell*wc_alpha_ell )
  
		wc_beta_ell = jnp.dot(self.A_ell, primal_sol_z.T).T-z_ell#*jnp.ones((self.num_batch, self.num_agent*self.num)) 
  
		r_ell = jnp.sqrt((wc_alpha_ell**2/a_ell**2)+(ws_alpha_ell**2/b_ell**2)+(wc_beta_ell**2/c_ell**2) )
		beta_ell = jnp.arccos(wc_beta_ell/(c_ell*r_ell))
  
  
		################### valid only for spheroid. Need to change for ellipsoid	
		# c1_ell = 1.0*self.rho_ell* a_ell**2*jnp.sin(beta_ell)**2 + 1.0*self.rho_ell * c_ell**2*jnp.cos(beta_ell)**2
		# c2_ell = 1.0*(a_ell*self.rho_ell*wc_alpha_ell*jnp.cos(alpha_ell)*jnp.sin(beta_ell) + b_ell*self.rho_ell*ws_alpha_ell*jnp.sin(alpha_ell)*jnp.sin(beta_ell) + c_ell*self.rho_ell*wc_beta_ell*jnp.cos(beta_ell))

		# c1_ell =  1.0*a_ell**2*self.rho_ell*jnp.sin(beta_ell)**2*jnp.cos(alpha_ell)**2 + 2*b_ell**2*jnp.sin(alpha_ell)**2*jnp.sin(beta_ell)**2 + 2*c_ell**2*jnp.cos(beta_ell)**2	
		# c2_ell =  1.0*a_ell*self.rho_ell*wc_alpha_ell*jnp.sin(beta_ell)*jnp.cos(alpha_ell) + 2.0*b_ell*ws_alpha_ell*jnp.sin(alpha_ell)*jnp.sin(beta_ell) + 2.0*c_ell*wc_beta_ell*jnp.cos(beta_ell)

		# d_ell = c2_ell/c1_ell

		d_ell = r_ell

		d_ell = jnp.minimum(jnp.ones((self.num_batch,  self.num*(self.num_agent)   )), d_ell   )
  
		res_x_ell = wc_alpha_ell-(a_ell)*d_ell*jnp.cos(alpha_ell)*jnp.sin(beta_ell)
		res_y_ell = ws_alpha_ell-(b_ell)*d_ell*jnp.sin(alpha_ell)*jnp.sin(beta_ell)
		res_z_ell = wc_beta_ell-(c_ell)*d_ell*jnp.cos(beta_ell)

  
		########################################################################### 
  
		wc_alpha_v = jnp.dot(self.A_v, primal_sol_x.T).T 
		ws_alpha_v = jnp.dot(self.A_v, primal_sol_y.T).T 
		wc_beta_v = jnp.dot(self.A_v, primal_sol_z.T).T
  

		xdot = wc_alpha_v 
		ydot = ws_alpha_v 
		zdot = wc_beta_v

		alpha_v = jnp.arctan2(ws_alpha_v, wc_alpha_v)
  
		ws_beta_v = wc_alpha_v/jnp.cos(alpha_v)
		beta_v = jnp.arctan2( ws_beta_v, wc_beta_v)
  
		# alpha_v = jnp.clip(alpha_v, -jnp.pi/5*jnp.ones(( self.num_batch, self.num*self.num_agent   )), jnp.pi/5*jnp.ones(( self.num_batch, self.num*self.num_agent   ))    )
		c1_d_v = 1.0*self.rho_ineq#*(jnp.cos(alpha_v)**2 * jnp.sin(beta_v)**2 + jnp.sin(alpha_v)**2 * jnp.sin(beta_v)**2 + jnp.cos(beta_v)**2 )
		c2_d_v = 1.0*self.rho_ineq*(wc_alpha_v*jnp.cos(alpha_v)*jnp.sin(beta_v) + ws_alpha_v*jnp.sin(alpha_v)*jnp.sin(beta_v) +wc_beta_v *jnp.cos(beta_v) )

		d_v = c2_d_v/c1_d_v

		# d_v = jnp.minimum(self.v_max*jnp.ones((self.num_batch, self.num*self.num_agent)), d_v   )
  
		d_v = jnp.clip(d_v, jnp.zeros(( self.num_batch, self.num*self.num_agent  )),  self.v_max*jnp.ones((self.num_batch, self.num*self.num_agent))  )

	
		############################################################################################

		wc_alpha_a = jnp.dot(self.A_a, primal_sol_x.T).T 
		ws_alpha_a = jnp.dot(self.A_a, primal_sol_y.T).T
		wc_beta_a = jnp.dot(self.A_a, primal_sol_z.T).T

		xddot = wc_alpha_a 
		yddot = ws_alpha_a 
		zddot = wc_beta_a 
  
		alpha_a = jnp.arctan2(ws_alpha_a, wc_alpha_a)
		# wc_beta_a = zddot_guess
		ws_beta_a = wc_alpha_a/jnp.cos(alpha_a)
		beta_a = jnp.arctan2( ws_beta_a, wc_beta_a)

		c1_d_a = 1.0*self.rho_ineq#*(jnp.cos(alpha_a)**2 * jnp.sin(beta_a)**2 + jnp.sin(alpha_a)**2 * jnp.sin(beta_a)**2 + jnp.cos(beta_a)**2 )
		c2_d_a = 1.0*self.rho_ineq*(wc_alpha_a*jnp.cos(alpha_a)*jnp.sin(beta_a) + ws_alpha_a*jnp.sin(alpha_a)*jnp.sin(beta_a) +wc_beta_a *jnp.cos(beta_a) )

		d_a = c2_d_a/c1_d_a

		d_a = jnp.clip(d_a, jnp.zeros(( self.num_batch, self.num*self.num_agent  )),  self.a_max*jnp.ones((self.num_batch, self.num*self.num_agent))  )


		
		res_ax_vec = xddot-d_a*jnp.cos(alpha_a) * jnp.sin(beta_a)
		res_ay_vec = yddot-d_a*jnp.sin(alpha_a) * jnp.sin(beta_a)
		res_az_vec = zddot-d_a*jnp.cos(beta_a)

		res_vx_vec = xdot-d_v*jnp.cos(alpha_v) * jnp.sin(beta_v)
		res_vy_vec = ydot-d_v*jnp.sin(alpha_v) * jnp.sin(beta_v)
		res_vz_vec = zdot-d_v*jnp.cos(beta_v)
 
	
		lamda_x = lamda_x-self.rho_agent*jnp.dot(self.A_fc.T, res_x_agent.T).T-self.rho_ell*jnp.dot(self.A_ell.T, res_x_ell.T).T
		lamda_y = lamda_y-self.rho_agent*jnp.dot(self.A_fc.T, res_y_agent.T).T-self.rho_ell*jnp.dot(self.A_ell.T, res_y_ell.T).T
		lamda_z = lamda_z-self.rho_agent*jnp.dot(self.A_fc.T, res_z_agent.T).T-self.rho_ell*jnp.dot(self.A_ell.T, res_z_ell.T).T
			
		primal_residual = jnp.linalg.norm(res_x_agent, axis = 1)+jnp.linalg.norm(res_y_agent, axis = 1)+jnp.linalg.norm(res_z_agent, axis = 1)\
  						  +jnp.linalg.norm(res_x_ell, axis = 1)+jnp.linalg.norm(res_y_ell, axis = 1)+jnp.linalg.norm(res_z_ell, axis = 1)
	   
	   
		# lamda_x = lamda_x-self.rho_bounds*jnp.dot(self.A_bounds.T, res_bounds_x.T).T
		# lamda_y = lamda_y-self.rho_bounds*jnp.dot(self.A_bounds.T, res_bounds_y.T).T
		# lamda_z = lamda_z-self.rho_bounds*jnp.dot(self.A_bounds.T, res_bounds_z.T).T
			
		# primal_residual = jnp.linalg.norm(res_x_agent, axis = 1)+jnp.linalg.norm(res_y_agent, axis = 1)+jnp.linalg.norm(res_z_agent, axis = 1) \
		# 				 +jnp.linalg.norm(res_bounds_x, axis = 1)+jnp.linalg.norm(res_bounds_y, axis = 1)+jnp.linalg.norm(res_bounds_z, axis = 1)
	   
		# primal_residual = jnp.linalg.norm(res_bounds_x, axis = 1)+jnp.linalg.norm(res_bounds_y, axis = 1)+jnp.linalg.norm(res_bounds_z, axis = 1)
							


		return alpha_agent, beta_agent, d_agent, lamda_x, lamda_y, lamda_z, primal_residual, alpha_v, beta_v, d_v, alpha_a, beta_a, d_a, alpha_ell, beta_ell, d_ell



	@partial(jit, static_argnums=(0,))
	def solve(self, d_agent, alpha_agent, beta_agent, lamda_x, lamda_y, lamda_z, state_x, state_y, state_z, alpha_v, beta_v, d_v, alpha_a, beta_a, d_a, x_ell, y_ell, z_ell, a_ell, b_ell, c_ell, alpha_ell, beta_ell, d_ell):
	
		
		b_eq_x, b_eq_y, b_eq_z = self.compute_boundary_vec_batch(state_x, state_y, state_z )
		
  
  		# print(jnp.shape(b_eq_x)) 
		# kk

		alpha_agent_init = alpha_agent 
		beta_agent_init = beta_agent 
		d_agent_init = d_agent  
		lamda_x_init = lamda_x 
		lamda_y_init = lamda_y  
		lamda_z_init = lamda_z  
		primal_sol_x_init = jnp.zeros(( self.num_batch, self.nvar*self.num_agent   ))
		primal_sol_y_init = jnp.zeros(( self.num_batch, self.nvar*self.num_agent   ))
		primal_sol_z_init = jnp.zeros(( self.num_batch, self.nvar*self.num_agent   ))
		alpha_v_init = alpha_v 
		beta_v_init = beta_v 
		d_v_init = d_v 

		alpha_a_init = alpha_a 
		beta_a_init = beta_a 
		d_a_init = d_a 
		
		alpha_ell_init = alpha_ell 
		beta_ell_init = beta_ell 
		d_ell_init = d_ell
				
  
		# print(b_eq_x, b_eq_y, b_eq_z)
		# kk



		def lax_solve(carry, idx):
	  
			
			primal_sol_x, primal_sol_y, primal_sol_z, alpha_agent, beta_agent, d_agent, lamda_x, lamda_y, lamda_z, alpha_v, beta_v, d_v, alpha_a, beta_a, d_a, alpha_ell, beta_ell, d_ell  = carry	

			# alpha_agent_prev = alpha_agent 
			# beta_agent_prev = beta_agent 
			# d_agent_prev = d_agent
   
			primal_sol_x_prev = primal_sol_x  
			primal_sol_y_prev = primal_sol_y 
			primal_sol_z_prev = primal_sol_z 
   
			lamda_x_prev = lamda_x 
			lamda_y_prev = lamda_y 
			lamda_z_prev = lamda_z 
	  
	  
			primal_sol_x, primal_sol_y, primal_sol_z = self.compute_x(d_agent, alpha_agent, beta_agent, lamda_x, lamda_y, lamda_z, b_eq_x, b_eq_y, b_eq_z, alpha_v, beta_v, d_v, alpha_a, beta_a, d_a, x_ell, y_ell, z_ell, a_ell, b_ell, c_ell, alpha_ell, beta_ell, d_ell)
			alpha_agent, beta_agent, d_agent, lamda_x, lamda_y, lamda_z, primal_residual, alpha_v, beta_v, d_v, alpha_a, beta_a, d_a, alpha_ell, beta_ell, d_ell = self.compute_alpha_d(primal_sol_x, primal_sol_y, primal_sol_z, lamda_x, lamda_y, lamda_z, x_ell, y_ell, z_ell, a_ell, b_ell, c_ell)
			# dual_residual = self.compute_dual_residual(alpha_agent_prev, beta_agent_prev, d_agent_prev, alpha_agent, beta_agent, d_agent)
			
			res = primal_residual 
   
			dual_residual = jnp.linalg.norm( primal_sol_x_prev-primal_sol_x, axis = 1  )+jnp.linalg.norm( primal_sol_y_prev-primal_sol_y, axis = 1  )+jnp.linalg.norm( primal_sol_z_prev-primal_sol_z, axis = 1  )\
						   +jnp.linalg.norm(lamda_x_prev-lamda_x, axis = 1)+jnp.linalg.norm(lamda_y_prev-lamda_y, axis = 1)+jnp.linalg.norm(lamda_z_prev-lamda_z, axis = 1)
  

			return (primal_sol_x, primal_sol_y, primal_sol_z, alpha_agent, beta_agent, d_agent, lamda_x, lamda_y, lamda_z, alpha_v, beta_v, d_v, alpha_a, beta_a, d_a, alpha_ell, beta_ell, d_ell ), (res, dual_residual)
		 
		
		carry_init = ( primal_sol_x_init, primal_sol_y_init, primal_sol_z_init, alpha_agent_init, beta_agent_init, d_agent_init, lamda_x_init, lamda_y_init, lamda_z_init, alpha_v_init, beta_v_init, d_v_init, alpha_a_init, beta_a_init, d_a_init, alpha_ell_init, beta_ell_init, d_ell_init  )
		carry_final, res_tot = lax.scan(lax_solve, carry_init, jnp.arange(self.maxiter))

		primal_sol_x, primal_sol_y, primal_sol_z, alpha_agent, beta_agent, d_agent, lamda_x, lamda_y, lamda_z, alpha_v, beta_v, d_v, alpha_a, beta_a, d_a, s_bounds_x, s_bounds_y, s_bounds_z  = carry_final 
		res, dual_residual = res_tot

		return primal_sol_x, primal_sol_y, primal_sol_z, res, dual_residual
	
	@partial(jit, static_argnums=(0,))
	def custom_forward(self, state_x_test, state_y_test, state_z_test, x_ell, y_ell, z_ell, a_ell, b_ell, c_ell):
		alpha_agent = jnp.zeros((self.num_batch, self.num_con*self.num))
		beta_agent = jnp.zeros((self.num_batch, self.num_con*self.num))
		d_agent  = jnp.ones((self.num_batch, self.num_con*self.num))


		lamda_x = jnp.zeros((self.num_batch, self.num_agent*self.nvar))
		lamda_y = jnp.zeros((self.num_batch, self.num_agent*self.nvar))
		lamda_z = jnp.zeros((self.num_batch, self.num_agent*self.nvar))


		s_bounds_x =  jnp.zeros((self.num_batch, self.num_agent*self.num*2))
		s_bounds_y =  jnp.zeros((self.num_batch, self.num_agent*self.num*2))
		s_bounds_z =  jnp.zeros((self.num_batch, self.num_agent*self.num*2))

		alpha_v = jnp.zeros(( self.num_batch, self.num*self.num_agent   ))
		beta_v = jnp.zeros(( self.num_batch, self.num*self.num_agent   ))
		d_v = jnp.ones(( self.num_batch, self.num*self.num_agent   ))

		alpha_a = jnp.zeros(( self.num_batch, self.num*self.num_agent   ))
		beta_a = jnp.zeros(( self.num_batch, self.num*self.num_agent   ))
		d_a = jnp.ones(( self.num_batch, self.num*self.num_agent   ))

		alpha_ell = jnp.zeros(( self.num_batch, self.num*self.num_agent   ))
		beta_ell = jnp.zeros(( self.num_batch, self.num*self.num_agent   ))
		d_ell = jnp.ones(( self.num_batch, self.num*self.num_agent   ))

		return self.solve(d_agent, alpha_agent, beta_agent, lamda_x, lamda_y, lamda_z, state_x_test, state_y_test, state_z_test, alpha_v, beta_v, d_v, alpha_a, beta_a, d_a, x_ell, y_ell, z_ell, a_ell, b_ell, c_ell, alpha_ell, beta_ell, d_ell)


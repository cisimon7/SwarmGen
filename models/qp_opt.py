import torch 
import scipy


# class QPOptimizer:
	
# 	def __init__(self, P, Pdot, Pddot, num_batch, num_agent, device):
		
# 		self.device = device
# 		self.num_batch = num_batch
# 		self.num_agent = num_agent
# 		self.num_con = int(scipy.special.binom(self.num_agent,2))

# 		# P Matrices
# 		self.P = P.to(device)
# 		self.Pdot = Pdot.to(device)
# 		self.Pddot = Pddot.to(device)

# 		# No. of Variables per agent
# 		self.nvar = P.size(dim=1)
# 		self.num  = P.size(dim=0)

# 		self.a_agent = 0.4
# 		self.b_agent = 0.80
		
# 		# Parameters
# 		t_fin = 10
# 		self.rho_agent = 1.0
# 		self.weight_smoothness = 1

# 		self.maxiter = 10
# 		self.tot_time = torch.linspace(0, t_fin, self.num, device=device)
# 		self.A_projection  = torch.eye(self.nvar*self.num_agent, device = device)
		
# 		A_eq_init = self.get_A_eq_init()
# 		self.A_eq_init = A_eq_init
		
# 		Q = self.get_Q()
# 		Q_tilda_inv_init = self.get_Q_tilde_inv_init(A_eq_init, Q)
# 		self.Q_tilda_inv_init = Q_tilda_inv_init
  
# 		self.compute_boundary_vec_batch = torch.vmap(self.compute_boundary_vec_single, in_dims = (0, 0, 0 )  )


# 	def compute_boundary_vec_single(self, state_x, state_y, state_z):

# 		b_eq_x = state_x.reshape(6, self.num_agent).T
# 		b_eq_y = state_y.reshape(6, self.num_agent).T 
# 		b_eq_z = state_z.reshape(6, self.num_agent).T 
		

# 		b_eq_x = b_eq_x.reshape(self.num_agent*(6))
# 		b_eq_y = b_eq_y.reshape(self.num_agent*(6))
# 		b_eq_z = b_eq_z.reshape(self.num_agent*(6))

# 		return b_eq_x, b_eq_y, b_eq_z

# 	def get_Q(self):

# 		Q = torch.kron(torch.eye(self.num_agent, device = self.device), torch.mm(self.Pddot.T, self.Pddot)) 

# 		return Q
	
# 	def get_Q_tilde_inv_init(self, A_eq_init, Q):
		
# 		return torch.linalg.inv(torch.vstack((torch.hstack(( torch.mm(self.A_projection.T, self.A_projection) , A_eq_init.T)), 
# 									 torch.hstack((A_eq_init, torch.zeros((A_eq_init.size(dim = 0), A_eq_init.size(dim = 0)), device = self.device  ) )  ))))	
 
# 	def get_A_eq_init(self):

# 		return torch.kron(torch.eye(self.num_agent, device=self.device), torch.vstack([self.P[0], self.Pdot[0], self.Pddot[0], self.P[-1], self.Pdot[-1], self.Pddot[-1] ] ))


# 	def qp_layer_init(self, b_eq_x_init, b_eq_y_init, b_eq_z_init, c_x_pred, c_y_pred, c_z_pred):

# 		lincost_x = torch.zeros((self.num_batch, self.nvar*self.num_agent  ), device = self.device)-torch.mm(self.A_projection.T, c_x_pred.T).T
# 		lincost_y = torch.zeros((self.num_batch, self.nvar*self.num_agent  ), device = self.device)-torch.mm(self.A_projection.T, c_y_pred.T).T
# 		lincost_z = torch.zeros((self.num_batch, self.nvar*self.num_agent  ), device = self.device)-torch.mm(self.A_projection.T, c_z_pred.T).T
							
# 		sol_x = torch.mm(self.Q_tilda_inv_init, torch.hstack([-lincost_x, b_eq_x_init]).T).T
# 		sol_y = torch.mm(self.Q_tilda_inv_init, torch.hstack([-lincost_y, b_eq_y_init]).T).T
# 		sol_z = torch.mm(self.Q_tilda_inv_init, torch.hstack([-lincost_z, b_eq_z_init]).T).T
		
# 		primal_sol_x = sol_x[:, 0:self.nvar*self.num_agent]
# 		primal_sol_y = sol_y[:, 0:self.nvar*self.num_agent]
# 		primal_sol_z = sol_z[:, 0:self.nvar*self.num_agent]

# 		return primal_sol_x, primal_sol_y, primal_sol_z 

# 	def __call__(self, c_x_pred, c_y_pred, c_z_pred, state_x, state_y, state_z):
		
# 		b_eq_x_init, b_eq_y_init, b_eq_z_init = self.compute_boundary_vec_batch(state_x, state_y, state_z)

# 		primal_sol_x_init, primal_sol_y_init, primal_sol_z_init = self.qp_layer_init(b_eq_x_init, b_eq_y_init, b_eq_z_init, c_x_pred, c_y_pred, c_z_pred)
	 
# 		return primal_sol_x_init, primal_sol_y_init, primal_sol_z_init




class QPOptimizer:
	
	def __init__(self, P, Pdot, Pddot, num_batch, num_agent, device):
		
		self.device = device
		self.num_batch = num_batch
		self.num_agent = num_agent
		self.num_con = int(scipy.special.binom(self.num_agent,2))

		# P Matrices
		self.P = P.to(device)
		self.Pdot = Pdot.to(device)
		self.Pddot = Pddot.to(device)

		# No. of Variables per agent
		self.nvar = P.size(dim=1)
		self.num  = P.size(dim=0)

		self.a_agent = 0.4
		self.b_agent = 0.80
		
		# Parameters
		t_fin = 10
		self.rho_agent = 1.0
		self.weight_smoothness = 1

		self.maxiter = 10
		self.tot_time = torch.linspace(0, t_fin, self.num, device=device)
		self.A_projection  = torch.eye(self.nvar*self.num_agent, device = device)

		Q = self.get_Q()
		self.Q = Q

		A_fc = self.get_A_fc()
		self.A_fc = A_fc
		
		A_eq = self.get_A_eq()
		self.A_eq = A_eq
		
		A_eq_init = self.get_A_eq_init()
		self.A_eq_init = A_eq_init
		
		Q_tilda_inv_init = self.get_Q_tilde_inv_init(A_eq_init, Q)
		self.Q_tilda_inv_init = Q_tilda_inv_init
  
		Q_tilda_inv = self.get_Q_tilde_inv(A_fc, A_eq, Q)
		self.Q_tilda_inv = Q_tilda_inv
		
		self.compute_boundary_vec_batch = torch.vmap(self.compute_boundary_vec_single, in_dims = (0, 0, 0 )  )

		# self.num_via_xyz = 5
		self.num_via_xyz = 2
		self.num_via_points = 3 * self.num_via_xyz
		self.epsilon = 0.001
		self.num_eq = self.A_eq.size(dim = 0)

	def get_A_acc_mat(self):

		A_acc_mat = torch.kron(torch.eye(self.num_agent, device = self.device), self.Pddot) ################### smoothness cost of all agents stacked up

		return A_acc_mat

	def get_Q(self):

		Q = torch.kron(torch.eye(self.num_agent, device = self.device), torch.mm(self.Pddot.T, self.Pddot)) ################### smoothness cost of all agents stacked up

		return Q

	def get_A_fc(self):

		def get_A_r(i):
			tmp = torch.hstack((torch.zeros((i, self.num_agent-i), device = self.device), -torch.eye(i, device = self.device)) )
			tmp[:,self.num_agent-i-1]=1
			return tmp

		tmp = torch.vstack([get_A_r(i) for i in torch.arange(self.num_agent-1, 0, -1)])

		return torch.kron(tmp, self.P)

	def get_A_eq(self):

		return torch.kron(torch.eye(self.num_agent, device = self.device), torch.vstack([self.P[0], self.Pdot[0], self.Pddot[0], self.P[-1], self.Pdot[-1], self.Pddot[-1] ]))

	def get_Q_tilde_inv(self, A_fc, A_eq, Q):
		
		return torch.linalg.inv(torch.vstack((torch.hstack(( self.weight_smoothness*Q  + self.rho_agent*torch.mm(A_fc.T, A_fc), A_eq.T)), 
									 torch.hstack((A_eq, torch.zeros((A_eq.size(dim = 0), A_eq.size(dim = 0)), device = self.device  ) )  ))))	
	
	def get_Q_tilde_inv_init(self, A_eq_init, Q):
		
		return torch.linalg.inv(torch.vstack((torch.hstack(( torch.mm(self.A_projection.T, self.A_projection) , A_eq_init.T)), 
									 torch.hstack((A_eq_init, torch.zeros((A_eq_init.size(dim = 0), A_eq_init.size(dim = 0)), device = self.device  ) )  ))))	
 
	def get_A_eq_init(self):

		return torch.kron(torch.eye(self.num_agent, device=self.device), torch.vstack([self.P[0], self.Pdot[0], self.Pddot[0], self.P[-1], self.Pdot[-1], self.Pddot[-1] ] ))

	def compute_boundary_vec_single(self, state_x, state_y, state_z):

		b_eq_x = state_x.reshape(6, self.num_agent).T
		b_eq_y = state_y.reshape(6, self.num_agent).T 
		b_eq_z = state_z.reshape(6, self.num_agent).T 
		

		b_eq_x = b_eq_x.reshape(self.num_agent*(6))
		b_eq_y = b_eq_y.reshape(self.num_agent*(6))
		b_eq_z = b_eq_z.reshape(self.num_agent*(6))

		return b_eq_x, b_eq_y, b_eq_z

	def qp_layer_init(self, b_eq_x_init, b_eq_y_init, b_eq_z_init, c_x_pred, c_y_pred, c_z_pred):

		lincost_x = torch.zeros((self.num_batch, self.nvar*self.num_agent  ), device = self.device)-torch.mm(self.A_projection.T, c_x_pred.T).T
		lincost_y = torch.zeros((self.num_batch, self.nvar*self.num_agent  ), device = self.device)-torch.mm(self.A_projection.T, c_y_pred.T).T
		lincost_z = torch.zeros((self.num_batch, self.nvar*self.num_agent  ), device = self.device)-torch.mm(self.A_projection.T, c_z_pred.T).T
							
		sol_x = torch.mm(self.Q_tilda_inv_init, torch.hstack([-lincost_x, b_eq_x_init]).T).T
		sol_y = torch.mm(self.Q_tilda_inv_init, torch.hstack([-lincost_y, b_eq_y_init]).T).T
		sol_z = torch.mm(self.Q_tilda_inv_init, torch.hstack([-lincost_z, b_eq_z_init]).T).T
		
		primal_sol_x = sol_x[:, 0:self.nvar*self.num_agent]
		primal_sol_y = sol_y[:, 0:self.nvar*self.num_agent]
		primal_sol_z = sol_z[:, 0:self.nvar*self.num_agent]

		return primal_sol_x, primal_sol_y, primal_sol_z 

	def __call__(self, c_x_pred, c_y_pred, c_z_pred, state_x, state_y, state_z):
		
		b_eq_x_init, b_eq_y_init, b_eq_z_init = self.compute_boundary_vec_batch(state_x, state_y, state_z)

		primal_sol_x_init, primal_sol_y_init, primal_sol_z_init = self.qp_layer_init(b_eq_x_init, b_eq_y_init, b_eq_z_init, c_x_pred, c_y_pred, c_z_pred)
	 
		return primal_sol_x_init, primal_sol_y_init, primal_sol_z_init

	def compute_alpha_d(self, primal_sol_x, primal_sol_y, primal_sol_z):

		##############computing x

		wc_alpha_agent = torch.mm(self.A_fc, primal_sol_x.T).T ### xi-xj

		############computing y 

		ws_alpha_agent = torch.mm(self.A_fc, primal_sol_y.T).T ### yi-yj
		alpha_agent = torch.arctan2(ws_alpha_agent, wc_alpha_agent )

		############computing z
  
		wc_beta_agent = torch.mm(self.A_fc, primal_sol_z.T).T ### zi-zj
		
		r_agent = torch.sqrt((wc_alpha_agent**2/self.a_agent**2)+(ws_alpha_agent**2/self.a_agent**2)+(wc_beta_agent**2/self.b_agent**2)+self.epsilon )
		beta_agent = torch.arccos(wc_beta_agent/(self.b_agent*r_agent))
  
		c_1_d_agent = 1.0*self.a_agent**2*torch.sin(beta_agent)**2 - 1.0*self.b_agent**2*torch.sin(beta_agent)**2 + 1.0*self.b_agent**2
		c_2_d_agent = 1.0*self.a_agent*wc_alpha_agent*torch.sin(beta_agent)*torch.cos(alpha_agent) + 1.0*self.a_agent*ws_alpha_agent*torch.sin(alpha_agent)*torch.sin(beta_agent) + 1.0*self.b_agent*wc_beta_agent*torch.cos(beta_agent)

		d_agent_temp = c_2_d_agent/c_1_d_agent
  
		d_agent = torch.maximum(torch.ones((self.num_batch, self.num_con*(self.num) ), device = self.device), d_agent_temp )
  
		res_x_agent = wc_alpha_agent-(self.a_agent)*d_agent*torch.cos(alpha_agent)*torch.sin(beta_agent)
		res_y_agent = ws_alpha_agent-(self.a_agent)*d_agent*torch.sin(alpha_agent)*torch.sin(beta_agent)
		res_z_agent = wc_beta_agent-(self.b_agent)*d_agent*torch.cos(beta_agent)

		primal_residual = torch.linalg.norm(res_x_agent, dim = 1)+torch.linalg.norm(res_y_agent, dim = 1)+torch.linalg.norm(res_z_agent, dim = 1)

		return primal_residual


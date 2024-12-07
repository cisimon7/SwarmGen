import sys
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import scipy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LabelNetBound2(torch.nn.Module):
    def __init__(self, input_height=16, input_width=2, input_channel=3, output_channel=3, nf=16, num_layers=5, lheight=16, lwidth=11):
        super(LabelNetBound2, self).__init__()
        x_kernel = np.floor(lheight * np.arange(1, num_layers+1)/num_layers).astype(np.int8)
        x_kernel = x_kernel.tolist()

        y_kernel = np.floor(lwidth * np.arange(1, num_layers+1)/num_layers).astype(np.int8)
        y_kernel = y_kernel.tolist()

        self.main = torch.nn.Sequential(
            torch.nn.Conv2d(
                input_channel, nf, 
                (input_height + 1 - (input_height % 2), input_width + 1 - (input_width % 2)), 1, 
                (input_height//2, input_width//2), 
                bias=False
            ),
            torch.nn.BatchNorm2d(nf),
            torch.nn.LeakyReLU(True),

            torch.nn.Sequential(*[
                torch.nn.Sequential(
                    torch.nn.AdaptiveAvgPool2d((max(3, x_kernel[i]), max(3, y_kernel[i]))),
                    torch.nn.Conv2d(
                        nf, nf, 
                        (max(3, x_kernel[i] + 1 - (x_kernel[i] % 2)), max(3, y_kernel[i] + 1 - (y_kernel[i] % 2))), 1, 
                        (max(3, x_kernel[i])//2, max(3, y_kernel[i])//2), bias=False),
                    torch.nn.LeakyReLU(True),
                )
                for i in range(num_layers)
            ]),

            torch.nn.Conv2d(nf, output_channel, 1, 1, bias=False)
        )

    def forward(self, h):
        return self.main(h)
	


class InitModelCNN(nn.Module):
	def __init__(self, inp_channel, emb_dims, pcd_features, inp_dim, hidden_dim, out_dim, min_inp, max_inp, inp_mean, inp_std):
		super(InitModelCNN, self).__init__()

		ndf = 64
		lheight, lwdith, tdim = 10, 10, 6
		self.cond_net = LabelNetBound2()

		self.decoder = torch.nn.Sequential(
			# laten vector Z, ``lheight x lwidth x zdim``
			torch.nn.ConvTranspose2d(6, ndf*8, 17, 1, 17//2, bias=False),
			torch.nn.BatchNorm2d(ndf*8),
			#  torch.nn.LeakyReLU(True),
			torch.nn.Softplus(),
			torch.nn.Dropout2d(0.1),

			torch.nn.AdaptiveAvgPool2d((16, 11)),
			torch.nn.ConvTranspose2d(ndf*8, ndf*4, (17, 11), 1, (17//2, 5), bias=False), #8,6
			torch.nn.BatchNorm2d(ndf*4),
			# torch.nn.LeakyReLU(True),
			torch.nn.Softplus(),
			torch.nn.Dropout2d(0.1),

			torch.nn.AdaptiveAvgPool2d((16, 11)),
			torch.nn.ConvTranspose2d(ndf*4, ndf*2, (17, 11), 1, (17//2, 5), bias=False), #10,8
			torch.nn.BatchNorm2d(ndf*2),
			#  torch.nn.LeakyReLU(True),
			torch.nn.Softplus(),
			torch.nn.Dropout2d(0.1),

			# th.nn.AdaptiveAvgPool2d((12, 10)),
			torch.nn.ConvTranspose2d(ndf*2, tdim, (17, 11), 1, (17//2, 5), bias=False), #12,10
			torch.nn.BatchNorm2d(tdim),
			# torch.nn.LeakyReLU(True),
			torch.nn.Tanh(),

			torch.nn.Linear(11, 11),
			# LambdaModule(lambda x: 3 * x)
		)

		self.min_inp = min_inp 
		self.max_inp = max_inp 

		# Normalizing Constants
		self.inp_mean = inp_mean
		self.inp_std = inp_std

		self.nvar = 11
		self.num_agent = 16
		
		# RCL Loss
		self.rcl_loss = nn.MSELoss()

	def forward(self, inp, c_pred, ellipse):
		inp_norm = (inp - self.inp_mean) / self.inp_std

		c_pred = c_pred.reshape(-1, 3, 16, 11) 

		inp_feature = self.cond_net(inp_norm)
		
		inp_cat = torch.cat([inp_feature, c_pred], dim=1)
	
		inp_cat = torch.cat([
			inp_cat,
			ellipse.unsqueeze(dim=-2).repeat_interleave(6, -2).unsqueeze(dim=-2).repeat_interleave(16, -2)
		], dim=-1)
		neural_output_batch = self.decoder(inp_cat)

		c_guess, lamda_init = neural_output_batch.split([3, 3], dim=-3)
		c_guess = c_pred + c_guess
	
		lamda_init = lamda_init.reshape(-1, 3*16*11)
		c_guess = c_guess.reshape(-1, 3*16*11)

		lamda_x_init = lamda_init[:, 0 : self.nvar*self.num_agent] 
		lamda_y_init = lamda_init[:, self.nvar*self.num_agent : 2*self.nvar*self.num_agent]  
		lamda_z_init = lamda_init[:, 2*self.nvar*self.num_agent : 3*self.nvar*self.num_agent] 
  
		c_x_guess = c_guess[:, 0: self.nvar*self.num_agent]
		c_y_guess = c_guess[:, self.nvar*self.num_agent : 2*self.nvar*self.num_agent]
		c_z_guess = c_guess[:, 2*self.nvar*self.num_agent : 3*self.nvar*self.num_agent]

		return c_x_guess, c_y_guess, c_z_guess, lamda_x_init, lamda_y_init, lamda_z_init

	def mlp_loss(self, accumulated_res_primal, accumulated_res_fixed_point, primal_sol_x, primal_sol_y, primal_sol_z, c_pred):
	 
	
		primal_loss = (0.5 * (torch.mean(accumulated_res_primal)))
		fixed_point_loss = (0.5 * (torch.mean(accumulated_res_fixed_point)))

		c_x_pred = c_pred[:, 0: self.nvar*self.num_agent]
		c_y_pred = c_pred[:, self.nvar*self.num_agent : 2*self.nvar*self.num_agent]
		c_z_pred = c_pred[:, 2*self.nvar*self.num_agent : 3*self.nvar*self.num_agent]
  
		proj_x_loss = 0.5*self.rcl_loss(primal_sol_x, c_x_pred)
		proj_y_loss = 0.5*self.rcl_loss(primal_sol_y, c_y_pred)
		proj_z_loss = 0.5*self.rcl_loss(primal_sol_z, c_z_pred)
  
		proj_loss = proj_x_loss + proj_y_loss + proj_z_loss
		loss = fixed_point_loss + proj_loss
 
		return loss, fixed_point_loss, primal_loss, proj_loss

	


class PointNet(nn.Module):
	def __init__(self, inp_channel, emb_dims, output_channels=20):
		super(PointNet, self).__init__()
		self.conv1 = nn.Conv1d(inp_channel, 64, kernel_size=1, bias=False) # input_channel = 3
		self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
		self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
		self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
		self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(64)
		self.bn3 = nn.BatchNorm1d(64)
		self.bn4 = nn.BatchNorm1d(128)
		self.bn5 = nn.BatchNorm1d(emb_dims)
		self.linear1 = nn.Linear(emb_dims, 256, bias=False)
		self.bn6 = nn.BatchNorm1d(256)
		self.dp1 = nn.Dropout()
		self.linear2 = nn.Linear(256, output_channels)
	
	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.relu(self.bn4(self.conv4(x)))
		x = F.relu(self.bn5(self.conv5(x)))
		x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
		x = F.relu(self.bn6(self.linear1(x)))
		x = self.dp1(x)
		x = self.linear2(x)
		return x


class MLP(nn.Module):
	def __init__(self, inp_dim, hidden_dim, out_dim):
		super(MLP, self).__init__()
		
		# MC Dropout Architecture
		self.mlp = nn.Sequential(
			nn.Linear(inp_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.LeakyReLU(inplace=True),
			# nn.Softplus(),

			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.LeakyReLU(inplace=True),
			# nn.Softplus(),

			nn.Linear(hidden_dim, 256),
			nn.BatchNorm1d(256),
			nn.LeakyReLU(inplace=True),
			# nn.Softplus(),
			
			nn.Linear(256, out_dim),
		)
	
	def forward(self, x):
		out = self.mlp(x)
		return out
	



class InitModel(nn.Module):
	def __init__(self, inp_channel, emb_dims, pcd_features, inp_dim, hidden_dim, out_dim, min_inp, max_inp, inp_mean, inp_std, num_agent):
		super(InitModel, self).__init__()
		self.mlp = MLP(inp_dim, hidden_dim, out_dim)
		self.point_net = PointNet(inp_channel, emb_dims, output_channels=pcd_features)

		self.min_inp = min_inp 
		self.max_inp = max_inp 

		# Normalizing Constants
		self.inp_mean = inp_mean
		self.inp_std = inp_std

		self.nvar = 11
		self.num_agent = num_agent
		
		# RCL Loss
		self.rcl_loss = nn.MSELoss()

	def forward(self, inp, c_pred):
		inp_norm = (inp - self.inp_mean) / self.inp_std
		inp_feature = self.point_net(inp_norm)
		inp_cat = torch.cat([inp_feature, c_pred], dim = 1)
	
		neural_output_batch = self.mlp(inp_cat)
		lamda_init = neural_output_batch[:, 0 : 3*self.nvar*self.num_agent ]
		c_guess = neural_output_batch[:, 3*self.nvar*self.num_agent: 6*self.nvar*self.num_agent ] + c_pred[:, :-6]
	
		lamda_x_init = lamda_init[:, 0 : self.nvar*self.num_agent] 
		lamda_y_init = lamda_init[:, self.nvar*self.num_agent : 2*self.nvar*self.num_agent]  
		lamda_z_init = lamda_init[:, 2*self.nvar*self.num_agent : 3*self.nvar*self.num_agent] 
  
		c_x_guess = c_guess[:, 0: self.nvar*self.num_agent]
		c_y_guess = c_guess[:, self.nvar*self.num_agent : 2*self.nvar*self.num_agent]
		c_z_guess = c_guess[:, 2*self.nvar*self.num_agent : 3*self.nvar*self.num_agent]

		return c_x_guess, c_y_guess, c_z_guess, lamda_x_init, lamda_y_init, lamda_z_init

	def mlp_loss(self, accumulated_res_primal, accumulated_res_fixed_point, primal_sol_x, primal_sol_y, primal_sol_z, c_pred):
	
		primal_loss = (0.5 * (torch.mean(accumulated_res_primal)))
		fixed_point_loss = (0.5 * (torch.mean(accumulated_res_fixed_point)))

		c_x_pred = c_pred[:, 0: self.nvar*self.num_agent]
		c_y_pred = c_pred[:, self.nvar*self.num_agent : 2*self.nvar*self.num_agent]
		c_z_pred = c_pred[:, 2*self.nvar*self.num_agent : 3*self.nvar*self.num_agent]
  
		proj_x_loss = 0.5*self.rcl_loss(primal_sol_x, c_x_pred)
		proj_y_loss = 0.5*self.rcl_loss(primal_sol_y, c_y_pred)
		proj_z_loss = 0.5*self.rcl_loss(primal_sol_z, c_z_pred)
  
		proj_loss = proj_x_loss+proj_y_loss+proj_z_loss
	
		loss = fixed_point_loss + proj_loss
 
		return loss, fixed_point_loss, primal_loss, proj_loss


class InitModel0(nn.Module):
	def __init__(self, inp_channel, emb_dims, pcd_features, inp_dim, hidden_dim, out_dim, min_inp, max_inp, inp_mean, inp_std):
		super(InitModel0, self).__init__()
		self.mlp = MLP(inp_dim, hidden_dim, out_dim)
		self.point_net = PointNet(inp_channel, emb_dims, output_channels=pcd_features)

		self.min_inp = min_inp 
		self.max_inp = max_inp 

		# Normalizing Constants
		self.inp_mean = inp_mean
		self.inp_std = inp_std

		self.nvar = 11
		self.num_agent = 16
		
		# RCL Loss
		self.rcl_loss = nn.MSELoss()

	def forward(self, inp, c_pred):
		inp_norm = (inp - self.inp_mean) / self.inp_std
		inp_feature = self.point_net(inp_norm)
		
		inp_cat = torch.cat([inp_feature, c_pred], dim = 1)
	
		neural_output_batch = self.mlp(inp_cat)
	
		lamda_init = neural_output_batch[:, 0 : 3*self.nvar*self.num_agent ]

		c_guess = neural_output_batch[:, 3*self.nvar*self.num_agent: 6*self.nvar*self.num_agent ] + c_pred[:, :-6]
	
		lamda_x_init = lamda_init[:, 0 : self.nvar*self.num_agent] 
		lamda_y_init = lamda_init[:, self.nvar*self.num_agent : 2*self.nvar*self.num_agent]  
		lamda_z_init = lamda_init[:, 2*self.nvar*self.num_agent : 3*self.nvar*self.num_agent] 
  
		c_x_guess = c_guess[:, 0: self.nvar*self.num_agent]
		c_y_guess = c_guess[:, self.nvar*self.num_agent : 2*self.nvar*self.num_agent]
		c_z_guess = c_guess[:, 2*self.nvar*self.num_agent : 3*self.nvar*self.num_agent]

		return c_x_guess, c_y_guess, c_z_guess, lamda_x_init, lamda_y_init, lamda_z_init

	def mlp_loss(self, accumulated_res_primal, accumulated_res_fixed_point, primal_sol_x, primal_sol_y, primal_sol_z, c_pred):
	 
	
		primal_loss = (0.5 * (torch.mean(accumulated_res_primal)))
		# fixed_point_loss = 0.5 * (torch.mean(accumulated_res_fixed_point_norm))+0.5 * (torch.mean(accumulated_res_fixed_point))

		# primal_loss = 0.5 * (torch.mean(accumulated_res_primal))
		fixed_point_loss = (0.5 * (torch.mean(accumulated_res_fixed_point)))
		

		c_x_pred = c_pred[:, 0: self.nvar*self.num_agent]
		c_y_pred = c_pred[:, self.nvar*self.num_agent : 2*self.nvar*self.num_agent]
		c_z_pred = c_pred[:, 2*self.nvar*self.num_agent : 3*self.nvar*self.num_agent]
  
		proj_x_loss = 0.5*self.rcl_loss(primal_sol_x, c_x_pred)
		proj_y_loss = 0.5*self.rcl_loss(primal_sol_y, c_y_pred)
		proj_z_loss = 0.5*self.rcl_loss(primal_sol_z, c_z_pred)
  
		proj_loss = proj_x_loss+proj_y_loss+proj_z_loss
	
		# loss = fixed_point_loss+1*primal_loss#+0.01*proj_loss
		loss = fixed_point_loss + proj_loss
 
		return loss, fixed_point_loss, primal_loss, proj_loss


class mlp_sf_multi_agent:
	
	def __init__(self, P, Pdot, Pddot, num_batch, num_agent):
		
		self.num_batch = num_batch
		self.num_agent = num_agent
		self.num_con = int(scipy.special.binom(self.num_agent,2))

		# P Matrices
		self.P = P.to(device)
		self.Pdot = Pdot.to(device)
		self.Pddot = Pddot.to(device)

		# No. of Variables per agent
		self.nvar = P.size(dim = 1)
		self.num = P.size(dim = 0)
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
		self.tot_time = torch.linspace(0, t_fin, self.num, device=device)

		# self.A_via_points = torch.vstack([ self.P[29], self.P[59]  ])
		
		######## computing all the necessary matrices

		# Q = self.get_Q()
		# self.Q = Q

		self.A_projection  = torch.eye(self.nvar*self.num_agent, device = device)

		
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
		
		self.compute_boundary_vec_batch = torch.vmap(self.compute_boundary_vec_single, in_dims = (0, 0, 0 )  )

		# self.num_via_xyz = 5
		self.num_via_xyz = 2
		self.num_via_points = 3 * self.num_via_xyz
		self.epsilon = 0.000001
		self.num_eq = self.A_eq.size(dim = 0)
  
	def get_A_ell(self):

		return torch.kron(torch.eye(self.num_agent, device = device), self.P )

	def get_A_v(self):

		return torch.kron(torch.eye(self.num_agent, device = device), self.Pdot )

	def get_A_a(self):

		return torch.kron(torch.eye(self.num_agent, device = device), self.Pddot )

	def get_A_acc_mat(self):

		A_acc_mat = torch.kron(torch.eye(self.num_agent, device = device), self.Pddot) ################### smoothness cost of all agents stacked up

		return A_acc_mat

	def get_A_fc(self):

		def get_A_r(i):
			tmp = torch.hstack((torch.zeros((i, self.num_agent-i), device = device), -torch.eye(i, device = device)) )
			tmp[:,self.num_agent-i-1]=1
			return tmp

		tmp = torch.vstack([get_A_r(i) for i in torch.arange(self.num_agent-1, 0, -1)])

		return torch.kron(tmp, self.P)

	def get_A_eq(self):

		return torch.kron(torch.eye(self.num_agent, device = device), torch.vstack([self.P[0], self.Pdot[0], self.Pddot[0], self.P[-1], self.Pdot[-1], self.Pddot[-1] ]))

	def get_Q_tilde_inv(self, A_fc, A_eq):
	#  +self.rho_ineq*torch.mm(self.A_a.T, self.A_a)
	 
		Q = torch.linalg.inv(torch.vstack((torch.hstack(( torch.mm(self.A_projection.T, self.A_projection)  + self.rho_agent*torch.mm(A_fc.T, A_fc)+self.rho_ell*torch.mm(self.A_ell.T, self.A_ell), A_eq.T)), 
									 torch.hstack((A_eq, torch.zeros((A_eq.size(dim = 0), A_eq.size(dim = 0)), device = device  ) )  ))))	
  
		# Q = torch.linalg.inv(torch.vstack((torch.hstack(( torch.mm(self.A_projection.T, self.A_projection)  + self.rho_agent*torch.mm(A_fc.T, A_fc)+self.rho_ell*torch.mm(self.A_ell.T, self.A_ell), A_eq.T)), 
		# 							 torch.hstack((A_eq, torch.zeros((A_eq.size(dim = 0), A_eq.size(dim = 0)), device = device  ) )  ))))	
		
		return Q

	def get_Q_tilde_inv_init(self, A_eq_init):
		
		return torch.linalg.inv(torch.vstack((torch.hstack(( torch.mm(self.A_projection.T, self.A_projection) , A_eq_init.T)), 
									 torch.hstack((A_eq_init, torch.zeros((A_eq_init.size(dim = 0), A_eq_init.size(dim = 0)), device = device  ) )  ))))	
 
	def get_A_eq_init(self):

		return torch.kron(torch.eye(self.num_agent, device = device), torch.vstack([self.P[0], self.Pdot[0], self.Pddot[0], self.P[-1], self.Pdot[-1], self.Pddot[-1] ] ))

	def qp_layer_init(self, b_eq_x_init, b_eq_y_init, b_eq_z_init, c_x_pred, c_y_pred, c_z_pred):

		lincost_x = torch.zeros((self.num_batch, self.nvar*self.num_agent), device = device)-torch.mm(self.A_projection.T, c_x_pred.T).T
		lincost_y = torch.zeros((self.num_batch, self.nvar*self.num_agent), device = device)-torch.mm(self.A_projection.T, c_y_pred.T).T
		lincost_z = torch.zeros((self.num_batch, self.nvar*self.num_agent), device = device)-torch.mm(self.A_projection.T, c_z_pred.T).T
		
		sol_x = torch.mm(self.Q_tilda_inv_init, torch.hstack([-lincost_x, b_eq_x_init]).T).T
		sol_y = torch.mm(self.Q_tilda_inv_init, torch.hstack([-lincost_y, b_eq_y_init]).T).T
		sol_z = torch.mm(self.Q_tilda_inv_init, torch.hstack([-lincost_z, b_eq_z_init]).T).T
		
		primal_sol_x = sol_x[:, 0:self.nvar*self.num_agent]
		primal_sol_y = sol_y[:, 0:self.nvar*self.num_agent]
		primal_sol_z = sol_z[:, 0:self.nvar*self.num_agent]

		return primal_sol_x, primal_sol_y, primal_sol_z 

	def compute_boundary_vec_single(self, state_x, state_y, state_z):

		b_eq_x = state_x.reshape(6, self.num_agent).T
		b_eq_y = state_y.reshape(6, self.num_agent).T 
		b_eq_z = state_z.reshape(6, self.num_agent).T 
		

		b_eq_x = b_eq_x.reshape(self.num_agent*(6))
		b_eq_y = b_eq_y.reshape(self.num_agent*(6))
		b_eq_z = b_eq_z.reshape(self.num_agent*(6))

		return b_eq_x, b_eq_y, b_eq_z

	def compute_x(self, c_x_pred, c_y_pred, c_z_pred, d_agent, alpha_agent, beta_agent, lamda_x, lamda_y, lamda_z, b_eq_x, b_eq_y, b_eq_z,  x_ell, y_ell, z_ell, a_ell, b_ell, c_ell, alpha_ell, beta_ell, d_ell):

		b_agent_x = self.a_agent*d_agent*torch.cos(alpha_agent)*torch.sin(beta_agent)
		b_agent_y = self.a_agent*d_agent*torch.sin(alpha_agent)*torch.sin(beta_agent)
		b_agent_z = self.b_agent*d_agent*torch.cos(beta_agent)
  
		b_ell_x = x_ell + a_ell*d_ell*torch.cos(alpha_ell)*torch.sin(beta_ell)
		b_ell_y = y_ell + b_ell*d_ell*torch.sin(alpha_ell)*torch.sin(beta_ell)
		b_ell_z = z_ell + c_ell*d_ell*torch.cos(beta_ell)
		
		lincost_x = -lamda_x-torch.mm(self.A_projection.T, c_x_pred.T).T-self.rho_agent*torch.mm(self.A_fc.T, b_agent_x.T).T-self.rho_ell*torch.mm(self.A_ell.T, b_ell_x.T).T
		lincost_y = -lamda_y-torch.mm(self.A_projection.T, c_y_pred.T).T-self.rho_agent*torch.mm(self.A_fc.T, b_agent_y.T).T-self.rho_ell*torch.mm(self.A_ell.T, b_ell_y.T).T
		lincost_z = -lamda_z-torch.mm(self.A_projection.T, c_z_pred.T).T-self.rho_agent*torch.mm(self.A_fc.T, b_agent_z.T).T-self.rho_ell*torch.mm(self.A_ell.T, b_ell_z.T).T
  
		sol_x = torch.mm(self.Q_tilda_inv, torch.hstack((-lincost_x, b_eq_x)).T).T
		sol_y = torch.mm(self.Q_tilda_inv, torch.hstack((-lincost_y, b_eq_y)).T).T
		sol_z = torch.mm(self.Q_tilda_inv, torch.hstack((-lincost_z, b_eq_z)).T).T

		################## from here
		primal_sol_x = sol_x[:, 0:self.num_agent*self.nvar]
		primal_sol_y = sol_y[:, 0:self.num_agent*self.nvar]
		primal_sol_z = sol_z[:, 0:self.num_agent*self.nvar]
		
		return primal_sol_x, primal_sol_y, primal_sol_z 
		
	def compute_alpha_d(self, primal_sol_x, primal_sol_y, primal_sol_z, lamda_x, lamda_y, lamda_z, x_ell, y_ell, z_ell, a_ell, b_ell, c_ell):
	 
		##############computing xi-xj

		wc_alpha_agent = torch.mm(self.A_fc, primal_sol_x.T).T ### xi-xj
  
		ws_alpha_agent = torch.mm(self.A_fc, primal_sol_y.T).T ### yi-yj
		alpha_agent = torch.arctan2(ws_alpha_agent, wc_alpha_agent )
  
		wc_beta_agent = torch.mm(self.A_fc, primal_sol_z.T).T ### zi-zj

		# print(wc_alpha_agent.sum().item(), ws_alpha_agent.sum().item(), alpha_agent.sum().item(), wc_beta_agent.sum().item())

		r_agent = torch.sqrt((wc_alpha_agent**2/self.a_agent**2)+(ws_alpha_agent**2/self.a_agent**2)+(wc_beta_agent**2/self.b_agent**2) +self.epsilon)
		beta_agent = torch.arccos(wc_beta_agent/(self.b_agent*r_agent+self.epsilon))

		# c_1_d_agent = 1.0*self.a_agent**2*torch.sin(beta_agent)**2 - 1.0*self.b_agent**2*torch.sin(beta_agent)**2 + 1.0*self.b_agent**2
		# c_2_d_agent = 1.0*self.a_agent*wc_alpha_agent*torch.sin(beta_agent)*torch.cos(alpha_agent) + 1.0*self.a_agent*ws_alpha_agent*torch.sin(alpha_agent)*torch.sin(beta_agent) + 1.0*self.b_agent*wc_beta_agent*torch.cos(beta_agent)

		# d_agent_temp = c_2_d_agent/c_1_d_agent
  
		d_agent_temp = r_agent
		d_agent = torch.maximum(torch.ones((self.num_batch, self.num_con*(self.num) ), device = device), d_agent_temp )
  
	
		res_x_agent = wc_alpha_agent-(self.a_agent)*d_agent*torch.cos(alpha_agent)*torch.sin(beta_agent)
		res_y_agent = ws_alpha_agent-(self.a_agent)*d_agent*torch.sin(alpha_agent)*torch.sin(beta_agent)
		res_z_agent = wc_beta_agent-(self.b_agent)*d_agent*torch.cos(beta_agent)

		######################## Elliptical bounds
  
  
		wc_alpha_ell = torch.mm(self.A_ell, primal_sol_x.T).T-x_ell # *torch.ones((self.num_batch, self.num_agent*self.num), device = device) 
		ws_alpha_ell = torch.mm(self.A_ell, primal_sol_y.T).T-y_ell # *torch.ones((self.num_batch, self.num_agent*self.num), device = device) 
		alpha_ell = torch.arctan2(a_ell*ws_alpha_ell, b_ell*wc_alpha_ell )
  
		wc_beta_ell = torch.mm(self.A_ell, primal_sol_z.T).T-z_ell #*torch.ones((self.num_batch, self.num_agent*self.num), device = device) 
  
		r_ell = torch.sqrt((wc_alpha_ell**2/a_ell**2)+(ws_alpha_ell**2/b_ell**2)+(wc_beta_ell**2/c_ell**2)+self.epsilon )
		beta_ell = torch.arccos(wc_beta_ell/(c_ell*r_ell+self.epsilon))
  
  
		# c1_ell =  1.0*a_ell**2*self.rho_ell*torch.sin(beta_ell)**2*torch.cos(alpha_ell)**2 + 2*b_ell**2*torch.sin(alpha_ell)**2*torch.sin(beta_ell)**2 + 2*c_ell**2*torch.cos(beta_ell)**2	
		# c2_ell =  1.0*a_ell*self.rho_ell*wc_alpha_ell*torch.sin(beta_ell)*torch.cos(alpha_ell) + 2.0*b_ell*ws_alpha_ell*torch.sin(alpha_ell)*torch.sin(beta_ell) + 2.0*c_ell*wc_beta_ell*torch.cos(beta_ell)

		# d_ell = c2_ell/c1_ell
  
		d_ell = r_ell
		d_ell = torch.minimum(torch.ones((self.num_batch,  self.num*(self.num_agent)   ), device = device), d_ell   )
  
		res_x_ell = wc_alpha_ell-(a_ell)*d_ell*torch.cos(alpha_ell)*torch.sin(beta_ell)
		res_y_ell = ws_alpha_ell-(b_ell)*d_ell*torch.sin(alpha_ell)*torch.sin(beta_ell)
		res_z_ell = wc_beta_ell-(c_ell)*d_ell*torch.cos(beta_ell)

	
		lamda_x = lamda_x-self.rho_agent*torch.mm(self.A_fc.T, res_x_agent.T).T-self.rho_ell*torch.mm(self.A_ell.T, res_x_ell.T).T
		lamda_y = lamda_y-self.rho_agent*torch.mm(self.A_fc.T, res_y_agent.T).T-self.rho_ell*torch.mm(self.A_ell.T, res_y_ell.T).T
		lamda_z = lamda_z-self.rho_agent*torch.mm(self.A_fc.T, res_z_agent.T).T-self.rho_ell*torch.mm(self.A_ell.T, res_z_ell.T).T
  
  
  
			
		primal_residual = torch.linalg.norm(res_x_agent, dim = 1)+torch.linalg.norm(res_y_agent, dim = 1)+torch.linalg.norm(res_z_agent, dim = 1)\
						 +torch.linalg.norm(res_x_ell, dim = 1)+torch.linalg.norm(res_y_ell, dim = 1)+torch.linalg.norm(res_z_ell, dim = 1)
	  


		return alpha_agent, beta_agent, d_agent, lamda_x, lamda_y, lamda_z, primal_residual, alpha_ell, beta_ell, d_ell
	
	def custom_forward(self, primal_sol_x, primal_sol_y, primal_sol_z, lamda_x, lamda_y, lamda_z, state_x, state_y, state_z, c_x_pred, c_y_pred, c_z_pred, x_ell, y_ell, z_ell, a_ell, b_ell, c_ell):

		b_eq_x, b_eq_y, b_eq_z = self.compute_boundary_vec_batch(state_x, state_y, state_z)

		######### one iteration before the main loop

		# print(primal_sol_x.sum().item(), primal_sol_y.sum().item(), primal_sol_z.sum().item(), lamda_x.sum().item(), lamda_y.sum().item(), lamda_z.sum().item(), x_ell.sum().item(), y_ell.sum().item(), z_ell.sum().item(), a_ell.sum().item(), b_ell.sum().item(), c_ell.sum().item())
		alpha_agent, beta_agent, d_agent, lamda_x, lamda_y, lamda_z, primal_residual, alpha_ell, beta_ell, d_ell = self.compute_alpha_d(primal_sol_x, primal_sol_y, primal_sol_z, lamda_x, lamda_y, lamda_z, x_ell, y_ell, z_ell, a_ell, b_ell, c_ell)
		# print(alpha_agent.sum().item(), beta_agent.sum().item(), d_agent.sum().item(), lamda_x.sum().item(), lamda_y.sum().item(), lamda_z.sum().item(), primal_residual.sum().item(), alpha_ell.sum().item(), beta_ell.sum().item(), d_ell.sum().item())
		primal_sol_x, primal_sol_y, primal_sol_z  = self.compute_x(c_x_pred, c_y_pred, c_z_pred, d_agent, alpha_agent, beta_agent, lamda_x, lamda_y, lamda_z, b_eq_x, b_eq_y, b_eq_z,  x_ell, y_ell, z_ell, a_ell, b_ell, c_ell, alpha_ell, beta_ell, d_ell)

		# print(primal_sol_x.sum().item(), primal_sol_y.sum().item(), primal_sol_z.sum().item(), "torch")
		# print("-"*10)
		# print(b_eq_x_init.sum().item(), b_eq_y_init.sum().item(), b_eq_z_init.sum().item(), "torch")
		################################################################################################

		res_primal = []
		res_fixed_point = []

		for i in range(0, self.maxiter):
	  

			primal_sol_x_prev = primal_sol_x.clone()
			primal_sol_y_prev = primal_sol_y.clone()
			primal_sol_z_prev = primal_sol_z.clone()
			
			lamda_x_prev = lamda_x.clone()
			lamda_y_prev = lamda_y.clone()
			lamda_z_prev = lamda_z.clone()
			
		
			alpha_agent, beta_agent, d_agent, lamda_x, lamda_y, lamda_z, primal_residual, alpha_ell, beta_ell, d_ell = self.compute_alpha_d(primal_sol_x, primal_sol_y, primal_sol_z, lamda_x, lamda_y, lamda_z, x_ell, y_ell, z_ell, a_ell, b_ell, c_ell)
			primal_sol_x, primal_sol_y, primal_sol_z  = self.compute_x(c_x_pred, c_y_pred, c_z_pred, d_agent, alpha_agent, beta_agent, lamda_x, lamda_y, lamda_z, b_eq_x, b_eq_y, b_eq_z,  x_ell, y_ell, z_ell, a_ell, b_ell, c_ell, alpha_ell, beta_ell, d_ell)

			fixed_point_residual = (
				torch.linalg.norm(primal_sol_x_prev - primal_sol_x, dim = 1) +
				torch.linalg.norm(primal_sol_y_prev - primal_sol_y, dim = 1) +
				torch.linalg.norm(primal_sol_z_prev - primal_sol_z, dim = 1) +
				torch.linalg.norm(lamda_x_prev - lamda_x, dim = 1) +
				torch.linalg.norm(lamda_y_prev - lamda_y, dim = 1) +
				torch.linalg.norm(lamda_z_prev - lamda_z, dim = 1)
			)
			
			res_primal.append(primal_residual)
			res_fixed_point.append(fixed_point_residual)
   
		res_primal_stack = torch.stack(res_primal )
		res_fixed_point_stack = torch.stack(res_fixed_point )

		accumulated_res_primal = torch.sum(res_primal_stack, dim = 0)/self.maxiter
		accumulated_res_fixed_point = torch.sum(res_fixed_point_stack, dim = 0)/self.maxiter	

		return primal_sol_x, primal_sol_y, primal_sol_z, accumulated_res_primal, accumulated_res_fixed_point, res_primal_stack, res_fixed_point_stack

	def decoder_function(self, inp_norm, state_x, state_y, state_z, c_pred, x_ell, y_ell, z_ell, a_ell, b_ell, c_ell):


		inp_feature = self.point_net(inp_norm)
		
		inp_cat = torch.cat([inp_feature, c_pred], dim = 1)
	
		neural_output_batch = self.mlp(inp_cat)
	
		lamda_init = neural_output_batch[:, 0 : 3*self.nvar*self.num_agent ]
  
		c_guess = neural_output_batch[:, 3*self.nvar*self.num_agent: 6*self.nvar*self.num_agent ]
	
		lamda_x_init = lamda_init[:, 0 : self.nvar*self.num_agent] 
		lamda_y_init = lamda_init[:, self.nvar*self.num_agent : 2*self.nvar*self.num_agent]  
		lamda_z_init = lamda_init[:, 2*self.nvar*self.num_agent : 3*self.nvar*self.num_agent] 
  
  
		c_x_pred = c_pred[:, 0: self.nvar*self.num_agent]
		c_y_pred = c_pred[:, self.nvar*self.num_agent : 2*self.nvar*self.num_agent]
		c_z_pred = c_pred[:, 2*self.nvar*self.num_agent : 3*self.nvar*self.num_agent]
  
		c_x_guess = c_guess[:, 0: self.nvar*self.num_agent]
		c_y_guess = c_guess[:, self.nvar*self.num_agent : 2*self.nvar*self.num_agent]
		c_z_guess = c_guess[:, 2*self.nvar*self.num_agent : 3*self.nvar*self.num_agent]
  
		b_eq_x_init, b_eq_y_init, b_eq_z_init = self.compute_boundary_vec_batch(state_x, state_y, state_z)

		primal_sol_x_init, primal_sol_y_init, primal_sol_z_init = self.qp_layer_init(b_eq_x_init, b_eq_y_init, b_eq_z_init, c_x_guess, c_y_guess, c_z_guess)
		
		primal_sol_x, primal_sol_y, primal_sol_z, accumulated_res_primal, accumulated_res_fixed_point, res_primal_stack, res_fixed_point_stack = self.custom_forward(primal_sol_x_init, primal_sol_y_init, primal_sol_z_init, lamda_x_init, lamda_y_init, lamda_z_init, state_x, state_y, state_z, c_x_pred, c_y_pred, c_z_pred, x_ell, y_ell, z_ell, a_ell, b_ell, c_ell)
	 
		return primal_sol_x, primal_sol_y, primal_sol_z, accumulated_res_primal, accumulated_res_fixed_point, res_primal_stack, res_fixed_point_stack

	def __call__(self, c_x_pred, c_y_pred, c_z_pred, c_x_guess, c_y_guess, c_z_guess, lamda_x_init, lamda_y_init, lamda_z_init, state_x, state_y, state_z, x_ell, y_ell, z_ell, a_ell, b_ell, c_ell):
  
		
	
		b_eq_x_init, b_eq_y_init, b_eq_z_init = self.compute_boundary_vec_batch(state_x, state_y, state_z)
		primal_sol_x_init, primal_sol_y_init, primal_sol_z_init = self.qp_layer_init(b_eq_x_init, b_eq_y_init, b_eq_z_init, c_x_guess, c_y_guess, c_z_guess)

		primal_sol_x, primal_sol_y, primal_sol_z, accumulated_res_primal, accumulated_res_fixed_point, res_primal_stack, res_fixed_point_stack = self.custom_forward(primal_sol_x_init, primal_sol_y_init, primal_sol_z_init, lamda_x_init, lamda_y_init, lamda_z_init, state_x, state_y, state_z, c_x_pred, c_y_pred, c_z_pred, x_ell, y_ell, z_ell, a_ell, b_ell, c_ell)
		return primal_sol_x, primal_sol_y, primal_sol_z, accumulated_res_primal, accumulated_res_fixed_point, res_primal_stack, res_fixed_point_stack


	
	
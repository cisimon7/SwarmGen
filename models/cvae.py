import sys
import torch as th
import numpy as np
import torch.nn as nn 
from .qp_opt import QPOptimizer


# Prevents NaN by torch.log(0)
def torch_log(x):
	return th.log(th.clamp(x, min = 1e-10))


class Encoder(nn.Module):
	def __init__(self, z_dim, tdim, lheight, lwidth, nef, ndf):
		super(Encoder, self).__init__()

		self.encoder = th.nn.Sequential(
			# state size ``tdim x num_agent x num_point  --> 3 x 16 x 50``
			th.nn.Conv2d(tdim, nef, (16+1, 11), 1, (8, 5), bias=False),
			th.nn.BatchNorm2d(nef),
			th.nn.LeakyReLU(0.2, inplace=True),

			th.nn.AdaptiveAvgPool2d((10, 40)),
			th.nn.Conv2d(nef, nef, (10+1, 11), 1, (5, 5), bias=False),
			th.nn.BatchNorm2d(nef),
			th.nn.LeakyReLU(0.2, inplace=True),

			th.nn.AdaptiveAvgPool2d((8, 30)),
			th.nn.Conv2d(nef, nef, (8+1, 11), 1, (4, 5), bias=False),
			th.nn.BatchNorm2d(nef),
			th.nn.LeakyReLU(0.2, inplace=True),

			th.nn.AdaptiveAvgPool2d((4, 20)),
			th.nn.Conv2d(nef, nef, (4+1, 11), 1, (2, 5), bias=False),
			th.nn.BatchNorm2d(nef),
			th.nn.LeakyReLU(0.2, inplace=True),

			th.nn.AdaptiveAvgPool2d((lwidth, z_dim)),
			th.nn.Conv2d(nef, lheight, 1, 1, 0, bias=True),
			
			th.nn.Flatten()
		)
		
		# Mean and Variance
		zdim = lheight * lwidth * z_dim
		self.mu = nn.Linear(zdim, zdim)
		self.var = nn.Linear(zdim, zdim)
		
		self.softplus = nn.Softplus()
		
	def forward(self, x, condition):
		x = th.cat([x, condition], dim=-1)
		out = self.encoder(x)

		mu = self.mu(out)
		var = self.var(out)
		return mu, self.softplus(var)
	

class Decoder(nn.Module):
	def __init__(self, z_dim, tdim, lheight, lwidth, nef, ndf, num_agent, nvar=11):
		super(Decoder, self).__init__()
		self.lheight, self.lwidth, self.zdim = lheight, lwidth, z_dim

		self.decoder = th.nn.Sequential(
			# laten vector Z, ``lheight x lwidth x zdim``
			th.nn.ConvTranspose2d(lheight, ndf*8, (10, 2), 1, bias=False),
			th.nn.BatchNorm2d(ndf*8),
			th.nn.ReLU(True),
			th.nn.Dropout2d(0.1),

			th.nn.AdaptiveAvgPool2d((8, 6)),
			th.nn.ConvTranspose2d(ndf*8, ndf*4, (5, 2), 1, bias=False), #8,6
			th.nn.BatchNorm2d(ndf*4),
			th.nn.ReLU(True),
			th.nn.Dropout2d(0.1),

			th.nn.AdaptiveAvgPool2d((10, 8)),
			th.nn.ConvTranspose2d(ndf*4, ndf*2, (6, 5), 1, bias=False), #10,8
			th.nn.BatchNorm2d(ndf*2),
			th.nn.ReLU(True),
			th.nn.Dropout2d(0.1),

			th.nn.ConvTranspose2d(ndf*2, tdim, (5, 3), 1, bias=False), #12,10
			th.nn.BatchNorm2d(tdim),
			th.nn.ReLU(True),

			th.nn.AdaptiveAvgPool2d((num_agent, nvar)),
			th.nn.Linear(11, 11),
		)

		self.cond_net = th.nn.Sequential(
			th.nn.ConvTranspose2d(tdim, ndf, (5, 2), 1, bias=False),
			th.nn.BatchNorm2d(ndf),
			th.nn.ReLU(True),

			th.nn.AdaptiveAvgPool2d((lwidth, z_dim)),
			th.nn.Conv2d(ndf, lheight, 1, 1, 0, bias=True),

		)
	
	def forward(self, z, condition):
		z = z.reshape(-1, self.lheight, self.lwidth, self.zdim)
		condition = self.cond_net(condition)
		inp_features = th.cat([z, condition], dim=-1)
		out = self.decoder(inp_features)
		return out
	
	

class CVAEModel(th.nn.Module):
	def __init__(
			self, P, Pdot, Pddot, num_agent, nvar, 
			cond_mean, cond_std, zdim, tdim, 
			lheight, lwidth, nef, ndf, device="cuda", batch=None
		):
		super(CVAEModel, self).__init__()

		self.num_agent = num_agent
		self.nvar = nvar
		
		self.qp_layer = QPOptimizer(P, Pdot, Pddot, batch, num_agent, device)

		self.encoder = Encoder(zdim, tdim, lheight, lwidth, nef, ndf)
		self.decoder = Decoder(zdim, tdim, lheight, lwidth, nef, ndf, num_agent)
		self.cond_mean, self.cond_std = cond_mean, cond_std

	def reparametrize(self, mean, std):
		eps = th.randn_like(mean)
		return mean + std * eps
	
	def forward(self, gt_traj, condition, x_init, y_init, z_init, x_fin, y_fin, z_fin):
		mean, std = self.encode(gt_traj, condition)
		z = self.reparametrize(mean, std)
		rec = self.decode(z, condition, x_init, y_init, z_init, x_fin, y_fin, z_fin)
		return rec, mean, std

	def decode(self, z, condition, x_init, y_init, z_init, x_fin, y_fin, z_fin):
		rec_nvar = self.decoder(z, condition)
		
		cx, cy, cz = rec_nvar[:, 0, ...], rec_nvar[:, 1, ...], rec_nvar[:, 2, ...]

		state_x = th.cat([
			x_init, th.zeros_like(x_init), th.zeros_like(x_init), 
			x_fin, th.zeros_like(x_fin), th.zeros_like(x_fin)
		], dim=-1)

		state_y = th.cat([
			y_init, th.zeros_like(y_init), th.zeros_like(y_init), 
			y_fin, th.zeros_like(y_fin), th.zeros_like(y_fin)
		], dim=-1)

		state_z = th.cat([
			z_init, th.zeros_like(z_init), th.zeros_like(z_init), 
			z_fin, th.zeros_like(z_fin), th.zeros_like(z_fin)
		], dim=-1)

		cx, cy, cz = cx.flatten(start_dim=-2), cy.flatten(start_dim=-2), cz.flatten(start_dim=-2)

		primal_sol_x, primal_sol_y, primal_sol_z = self.qp_layer(cx, cy, cz, state_x, state_y, state_z)
		primal_sol_x, primal_sol_y, primal_sol_z = (
			primal_sol_x.reshape(-1, self.num_agent, self.nvar),
			primal_sol_y.reshape(-1, self.num_agent, self.nvar),
			primal_sol_z.reshape(-1, self.num_agent, self.nvar)
		)

		return th.stack([primal_sol_x, primal_sol_y, primal_sol_z], dim=1)
	
	def encode(self, gt_traj, condition):
		condition = condition - self.cond_mean / self.cond_std
		mean, std = self.encoder(gt_traj, condition)        
  
		return mean, std  

	
class LabelNet(th.nn.Module):
    def __init__(self, input_channel=3, nf=10):
        super(LabelNet, self).__init__()
        self.main = th.nn.Sequential(
            th.nn.Conv2d(input_channel, nf, (5, 1), (2, 1), (0, 2), bias=False),
			th.nn.BatchNorm2d(nf),
			th.nn.LeakyReLU(0.2, inplace=True),

            th.nn.Conv2d(nf, 3, (3, 2), 1, (0, 0), bias=False),
			th.nn.BatchNorm2d(3),
			th.nn.LeakyReLU(0.2, inplace=True),

			th.nn.Linear(5, 64),
			th.nn.BatchNorm2d(3),
			th.nn.ReLU(True),

			th.nn.Linear(64, 5)
        )

    def forward(self, h):
        return self.main(h)
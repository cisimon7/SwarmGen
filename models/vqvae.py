import torch as th
from .utils import weights_init
from .qp_opt import QPOptimizer


class VectorQuantizerLayer(th.nn.Module):
	def __init__(self, num_embeddings, embedding_dim, beta=0.2):
		super(VectorQuantizerLayer, self).__init__()
		self.embedding_dim = embedding_dim
		self.num_embeddings = num_embeddings

		self.embedding = th.nn.Embedding(self.num_embeddings, self.embedding_dim, dtype=th.float32)
		# th.nn.init.uniform_(self.embedding.weight, -1/self.num_embeddings, 1/self.num_embeddings)
		th.nn.init.uniform_(self.embedding.weight, -2, 2)

		self.beta = beta
		
	def get_code_indices(self, x):
		x_flat = x.reshape(-1, self.embedding_dim)
		distances = th.cdist(x_flat, self.embedding.weight)
		distances = distances.argmin(dim=-1, keepdim=True)
		return distances.reshape(x.size()[:-1])

	def forward(self, x: th.Tensor):
		indices = self.get_code_indices(x)
		quantized = self.embedding(indices)
		
		commitment_loss = th.nn.functional.mse_loss(quantized.detach(), x)
		codebook_loss = th.nn.functional.mse_loss(quantized, x.detach())
		
		loss = (self.beta * commitment_loss) + codebook_loss
		quantized = x + (quantized - x).detach()
		
		return quantized, indices, loss
	

class VQVAEAgents(th.nn.Module):
	def __init__(
			self, P, Pdot, Pddot, num_agent, nvar, emb_num: int, emb_dim: int, 
			zdim: int, tdim: int, lheight: int, lwidth: int, 
			nef: int, ndf: int, beta=0.2, device="cuda", batch=None
	):
		super(VQVAEAgents, self).__init__()

		self.num_agent = num_agent
		self.nvar = nvar
		
		self.qp_layer = QPOptimizer(P, Pdot, Pddot, batch, num_agent, device)
		
		self.quantizer = VectorQuantizerLayer(emb_num, emb_dim, beta)

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

			th.nn.AdaptiveAvgPool2d((lwidth, zdim)),
			th.nn.Conv2d(nef, lheight, 1, 1, 0, bias=True),
			# state size ``lheight x lwidth x zdim``
		)

		self.decoder = th.nn.Sequential(
			# laten vector Z, ``lheight x lwidth x zdim``
			th.nn.ConvTranspose2d(lheight, ndf*8, (10, 2), 1, bias=False),
			th.nn.BatchNorm2d(ndf*8),
			th.nn.ReLU(True),
			th.nn.Dropout2d(0.1),

			# th.nn.AdaptiveAvgPool2d((8, 6)),
			th.nn.ConvTranspose2d(ndf*8, ndf*4, (5, 2), 1, bias=False), #8,6
			th.nn.BatchNorm2d(ndf*4),
			th.nn.ReLU(True),
			th.nn.Dropout2d(0.1),

			# th.nn.AdaptiveAvgPool2d((10, 8)),
			th.nn.ConvTranspose2d(ndf*4, ndf*2, (6, 5), 1, bias=False), #10,8
			th.nn.BatchNorm2d(ndf*2),
			th.nn.ReLU(True),
			th.nn.Dropout2d(0.1),

			# # th.nn.AdaptiveAvgPool2d((12, 10)),
			th.nn.ConvTranspose2d(ndf*2, tdim, (5, 3), 1, bias=False), #12,10
			th.nn.BatchNorm2d(tdim),
			th.nn.ReLU(True),

			th.nn.AdaptiveAvgPool2d((num_agent, nvar)),
			th.nn.Linear(11, 11),
			
			# # state size. ``tdim x num_agent x nvar``
		)
		self.encoder.apply(weights_init)
		self.decoder.apply(weights_init)
		
		self.device = device
		self.to(device=device)

	def encode(self, target):
		enc_out = self.encoder(target)
		quantized_out, indices, quantizer_loss = self.quantizer(enc_out)

		return quantized_out, indices, quantizer_loss, enc_out
	
	def decode(self, x, x_init, y_init, z_init, x_fin, y_fin, z_fin):
		rec_nvar = self.decoder(x)
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
		
	def forward(self, gt_traj, x_init, y_init, z_init, x_fin, y_fin, z_fin):
		quantized_out, indices, quantizer_loss, enc_out = self.encode(gt_traj)
		rec = self.decode(quantized_out, x_init, y_init, z_init, x_fin, y_fin, z_fin)
		
		return rec, quantizer_loss	    
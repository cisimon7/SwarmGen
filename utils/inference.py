import numpy as np
import torch as th
from torch import Tensor

import jax
import jax.numpy as jnp
from time import time, perf_counter

from models.cvae import CVAEModel
from models.vqvae import VQVAEAgents
from models.sampler.pixelcnn import PixelCNN
from optimizers.mlp_sf_multi_agent_varying_bounds_1 import InitModel


def make_state(x_init, y_init, z_init, x_fin, y_fin, z_fin):

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

	return state_x, state_y, state_z


def make_inp(x_init, y_init, z_init, x_fin, y_fin, z_fin):
	return th.stack([
		x_init, y_init, z_init, 
		th.zeros_like(x_init), th.zeros_like(y_init), th.zeros_like(z_init), 
		th.zeros_like(x_init), th.zeros_like(y_init), th.zeros_like(z_init), 
		x_fin, y_fin, z_fin, 
		th.zeros_like(x_fin), th.zeros_like(y_fin), th.zeros_like(z_fin), 
		th.zeros_like(x_fin), th.zeros_like(y_fin), th.zeros_like(z_fin), 
	], dim=-2)


def run_pipeline(
		x_init, y_init, z_init, x_fin, y_fin, z_fin, center, radii,
		P: Tensor, vq_model: VQVAEAgents, sampler: PixelCNN, model_sf: InitModel, 
		multi_agent_learned, multi_agent_base, 
		num_agent, num=50, nvar=11, batch_size=10, lheight=3, lwidth=3, device="cuda", verbose=True, comp_type="base"
	):
	x_init, y_init, z_init, x_fin, y_fin, z_fin, center, radii = (
		x_init.to(device=device), y_init.to(device=device), z_init.to(device=device), 
		x_fin.to(device=device), y_fin.to(device=device), z_fin.to(device=device),
		center.to(device), radii.to(device) 
	)

	inits = th.stack([x_init, y_init, z_init], dim=-2)
	fins = th.stack([x_fin, y_fin, z_fin], dim=-2)
	conditions = th.stack([inits, fins], dim=-1)

	state_x, state_y, state_z = make_state(
		x_init, y_init, z_init, x_fin, y_fin, z_fin
	)
	inp = make_inp(x_init, y_init, z_init, x_fin, y_fin, z_fin)

	with th.no_grad():
		start = perf_counter()
		latent_idx = th.zeros(batch_size, num_agent, lheight, lwidth, dtype=th.float32, device=device)
		for i in range(lheight):
			for j in range(lwidth):
				output = sampler(latent_idx, conditions.permute(0, 2, 1, 3))
				latent_idx[:, :, i, j] = th.multinomial(
					th.nn.functional.softmax(output[:, :, i, j].cpu(), dim=-1),
					num_samples=1
				)

		latent = vq_model.quantizer.embedding(latent_idx.to(dtype=th.int32))[:, 0]

		if verbose:
			print(f"{'PixelCNN sampling time':<30}: {(perf_counter() - start):.4f} secs")
		
		start = perf_counter()
		c_pred = vq_model.decode(
			latent, 
			x_init.repeat_interleave(batch_size, 0), y_init.repeat_interleave(batch_size, 0), z_init.repeat_interleave(batch_size, 0), 
			x_fin.repeat_interleave(batch_size, 0), y_fin.repeat_interleave(batch_size, 0), z_fin.repeat_interleave(batch_size, 0)
		)
		if verbose:
			print(f"{'VQ decoder time':<30}: {(perf_counter() - start):.4f} secs")

		cx_pred, cy_pred, cz_pred = (
			c_pred[:, 0, ...].flatten(start_dim=-2), 
			c_pred[:, 1, ...].flatten(start_dim=-2), 
			c_pred[:, 2, ...].flatten(start_dim=-2)
		)
		c_pred_test = th.cat([cx_pred, cy_pred, cz_pred], dim=-1)

		### End - vq-px

		inp_test = th.vstack([inp] * batch_size)
		cen_test = th.vstack([center] * batch_size)
		rad_test = th.vstack([radii] * batch_size)
		state_x_test = th.vstack([state_x]*batch_size)
		state_y_test = th.vstack([state_y]*batch_size)
		state_z_test = th.vstack([state_z]*batch_size)

		c_x_guess, c_y_guess, c_z_guess, lamda_x, lamda_y, lamda_z = model_sf(inp_test, th.cat([c_pred_test, cen_test, rad_test], dim=-1))
		
		c_x_pred = c_pred_test[:, 0 : multi_agent_learned.nvar*multi_agent_learned.num_agent] 
		c_y_pred = c_pred_test[:, multi_agent_learned.nvar*multi_agent_learned.num_agent : 2*multi_agent_learned.nvar*multi_agent_learned.num_agent]  
		c_z_pred = c_pred_test[:, 2*multi_agent_learned.nvar*multi_agent_learned.num_agent : 3*multi_agent_learned.nvar*multi_agent_learned.num_agent]


		x_ell, y_ell, z_ell = center.permute(1, 0)
		x_ell, y_ell, z_ell = (
			x_ell[..., None].cpu() * th.ones((batch_size, num * num_agent)),
			y_ell[..., None].cpu() * th.ones((batch_size, num * num_agent)),
			z_ell[..., None].cpu() * th.ones((batch_size, num * num_agent))
		)

		a_ell, b_ell, c_ell = radii.permute(1, 0)
		a_ell, b_ell, c_ell = (
			a_ell[..., None].cpu() * th.ones((batch_size, num * num_agent)),
			b_ell[..., None].cpu() * th.ones((batch_size, num * num_agent)),
			c_ell[..., None].cpu() * th.ones((batch_size, num * num_agent))
		)
		
		lamda_x_jnp = jnp.asarray( lamda_x.cpu().detach().numpy()  )
		lamda_y_jnp = jnp.asarray( lamda_y.cpu().detach().numpy()  )
		lamda_z_jnp = jnp.asarray( lamda_z.cpu().detach().numpy()  )
		
		c_x_pred_jnp = jnp.asarray( c_x_pred.cpu().detach().numpy()  )
		c_y_pred_jnp = jnp.asarray( c_y_pred.cpu().detach().numpy()  )
		c_z_pred_jnp = jnp.asarray( c_z_pred.cpu().detach().numpy()  )
		
		c_x_guess_jnp = jnp.asarray( c_x_guess.cpu().detach().numpy()  )
		c_y_guess_jnp = jnp.asarray( c_y_guess.cpu().detach().numpy()  )
		c_z_guess_jnp = jnp.asarray( c_z_guess.cpu().detach().numpy()  )

		c_x_guess = th.from_numpy(np.array(c_x_guess_jnp))
		c_y_guess = th.from_numpy(np.array(c_y_guess_jnp))
		c_z_guess = th.from_numpy(np.array(c_z_guess_jnp))
			
		state_x_test_jnp = jnp.asarray(state_x_test.cpu().detach().numpy())
		state_y_test_jnp = jnp.asarray(state_y_test.cpu().detach().numpy())
		state_z_test_jnp = jnp.asarray(state_z_test.cpu().detach().numpy())

		start = perf_counter()
		primal_sol_x, primal_sol_y, primal_sol_z, accumulated_res_primal, accumulated_res_fixed_point, primal_residual, dual_residual = jax.block_until_ready(multi_agent_learned(
			c_x_pred_jnp, c_y_pred_jnp, c_z_pred_jnp, c_x_guess_jnp, c_y_guess_jnp, c_z_guess_jnp, 
			lamda_x_jnp, lamda_y_jnp, lamda_z_jnp, state_x_test_jnp, state_y_test_jnp, state_z_test_jnp, 
			x_ell.cpu().numpy(), y_ell.cpu().numpy(), z_ell.cpu().numpy(), a_ell.cpu().numpy(), b_ell.cpu().numpy(), c_ell.cpu().numpy()
		))

		if verbose:
			print(f"{'Learned optimizer time':<30}: {(perf_counter() - start):.4f} secs")

		start = perf_counter()
		if comp_type == "base":
			primal_sol_x_base, primal_sol_y_base, primal_sol_z_base, primal_residual_base, dual_residual_base = multi_agent_base.custom_forward(
				state_x_test_jnp, state_y_test_jnp, state_z_test_jnp, 
				x_ell.cpu().numpy(), y_ell.cpu().numpy(), z_ell.cpu().numpy(), 
				a_ell.cpu().numpy(), b_ell.cpu().numpy(), c_ell.cpu().numpy()
			)
		elif comp_type == "sf_cpred":
			primal_sol_x_base, primal_sol_y_base, primal_sol_z_base, _, _, primal_residual_base, dual_residual_base = jax.block_until_ready(multi_agent_learned(
				c_x_pred_jnp, c_y_pred_jnp, c_z_pred_jnp, 
				c_x_pred_jnp, c_y_pred_jnp, c_z_pred_jnp, 
				jnp.zeros_like(lamda_x_jnp), jnp.zeros_like(lamda_y_jnp), jnp.zeros_like(lamda_z_jnp), 
				state_x_test_jnp, state_y_test_jnp, state_z_test_jnp, 
				x_ell.cpu().numpy(), y_ell.cpu().numpy(), z_ell.cpu().numpy(), 
				a_ell.cpu().numpy(), b_ell.cpu().numpy(), c_ell.cpu().numpy()
			))
		elif comp_type == "sf_zero":
			primal_sol_x_base, primal_sol_y_base, primal_sol_z_base, _, _, primal_residual_base, dual_residual_base = jax.block_until_ready(multi_agent_learned(
				c_x_pred_jnp, c_y_pred_jnp, c_z_pred_jnp, 
				jnp.zeros_like(c_x_pred_jnp), jnp.zeros_like(c_y_pred_jnp), jnp.zeros_like(c_z_pred_jnp), 
				jnp.zeros_like(lamda_x_jnp), jnp.zeros_like(lamda_y_jnp), jnp.zeros_like(lamda_z_jnp), 
				state_x_test_jnp, state_y_test_jnp, state_z_test_jnp, 
				x_ell.cpu().numpy(), y_ell.cpu().numpy(), z_ell.cpu().numpy(), 
				a_ell.cpu().numpy(), b_ell.cpu().numpy(), c_ell.cpu().numpy()
			))
		else:
			raise TypeError(f"{comp_type} is not recoginised")

		if verbose:
			print(f"{'Base/Comp optimizer time':<30}: {(perf_counter() - start):.4f} secs")

		try:
			best_idx = np.nanargmin(primal_residual[-1, :] + 0*dual_residual[-1, :])
			best_idx_base = np.nanargmin(primal_residual_base[-1, :] + 0*dual_residual_base[-1, :])
		except ValueError:
			raise 

		primal_sol_x_1 = primal_sol_x[best_idx]
		primal_sol_y_1 = primal_sol_y[best_idx]
		primal_sol_z_1 = primal_sol_z[best_idx]

		primal_sol_x_1 = primal_sol_x_1.reshape(multi_agent_learned.num_agent, multi_agent_learned.nvar)
		primal_sol_y_1 = primal_sol_y_1.reshape(multi_agent_learned.num_agent, multi_agent_learned.nvar)
		primal_sol_z_1 = primal_sol_z_1.reshape(multi_agent_learned.num_agent, multi_agent_learned.nvar)

		primal_sol_x_1_base = primal_sol_x_base[best_idx_base]
		primal_sol_y_1_base = primal_sol_y_base[best_idx_base]
		primal_sol_z_1_base = primal_sol_z_base[best_idx_base]

		primal_sol_x_1_base = primal_sol_x_1_base.reshape(multi_agent_learned.num_agent, multi_agent_learned.nvar)
		primal_sol_y_1_base = primal_sol_y_1_base.reshape(multi_agent_learned.num_agent, multi_agent_learned.nvar)
		primal_sol_z_1_base = primal_sol_z_1_base.reshape(multi_agent_learned.num_agent, multi_agent_learned.nvar)
		
		primal_sol_x_1_init = c_x_pred_jnp[best_idx]
		primal_sol_y_1_init = c_y_pred_jnp[best_idx]
		primal_sol_z_1_init = c_z_pred_jnp[best_idx]

		primal_sol_x_1_init = primal_sol_x_1_init.reshape(multi_agent_learned.num_agent, multi_agent_learned.nvar)
		primal_sol_y_1_init = primal_sol_y_1_init.reshape(multi_agent_learned.num_agent, multi_agent_learned.nvar)
		primal_sol_z_1_init = primal_sol_z_1_init.reshape(multi_agent_learned.num_agent, multi_agent_learned.nvar)

		x_traj = jnp.dot(multi_agent_learned.P, primal_sol_x_1.T).T
		y_traj = jnp.dot(multi_agent_learned.P, primal_sol_y_1.T).T
		z_traj = jnp.dot(multi_agent_learned.P, primal_sol_z_1.T).T

		x_traj_base = jnp.dot(multi_agent_learned.P, primal_sol_x_1_base.T).T
		y_traj_base = jnp.dot(multi_agent_learned.P, primal_sol_y_1_base.T).T
		z_traj_base = jnp.dot(multi_agent_learned.P, primal_sol_z_1_base.T).T
		
		x_traj_init = jnp.dot(multi_agent_learned.P, primal_sol_x_1_init.T).T
		y_traj_init = jnp.dot(multi_agent_learned.P, primal_sol_y_1_init.T).T
		z_traj_init = jnp.dot(multi_agent_learned.P, primal_sol_z_1_init.T).T

		x_ddot = jnp.dot(multi_agent_learned.Pddot, primal_sol_x[best_idx].reshape(num_agent, nvar).T).T
		y_ddot = jnp.dot(multi_agent_learned.Pddot, primal_sol_y[best_idx].reshape(num_agent, nvar).T).T
		z_ddot = jnp.dot(multi_agent_learned.Pddot, primal_sol_z[best_idx].reshape(num_agent, nvar).T).T

		x_ddot_base = jnp.dot(multi_agent_learned.Pddot, primal_sol_x_base[best_idx_base].reshape(num_agent, nvar).T).T
		y_ddot_base = jnp.dot(multi_agent_learned.Pddot, primal_sol_y_base[best_idx_base].reshape(num_agent, nvar).T).T
		z_ddot_base = jnp.dot(multi_agent_learned.Pddot, primal_sol_z_base[best_idx_base].reshape(num_agent, nvar).T).T

		ddot = jnp.stack([x_ddot, y_ddot, z_ddot], axis=-1)
		mean_smoothness = jnp.linalg.norm(ddot, axis=-1).sum(axis=-1).mean()

		ddot_base = jnp.stack([x_ddot_base, y_ddot_base, z_ddot_base], axis=-1)
		mean_smoothness_base = jnp.linalg.norm(ddot_base, axis=-1).sum(axis=-1).mean()

		return (
			primal_residual[:, best_idx], primal_residual_base[:, best_idx_base], 
			dual_residual[:, best_idx], dual_residual_base[:, best_idx_base]
		), (
			x_traj, y_traj, z_traj,
			x_traj_base, y_traj_base, z_traj_base,
			x_traj_init, y_traj_init, z_traj_init,
		), (mean_smoothness, mean_smoothness_base)
	

def run_pipeline_cvae(
		x_init, y_init, z_init, x_fin, y_fin, z_fin, center, radii,
		zdim: int, P: Tensor, cv_model: CVAEModel, model_sf: InitModel, 
		multi_agent_learned, multi_agent_base, 
		num_agent, num=50, nvar=11, batch_size=10, lheight=3, lwidth=3, device="cuda", verbose=True, comp_type="base"
	):
	x_init, y_init, z_init, x_fin, y_fin, z_fin, center, radii = (
		x_init.to(device=device), y_init.to(device=device), z_init.to(device=device), 
		x_fin.to(device=device), y_fin.to(device=device), z_fin.to(device=device),
		center.to(device), radii.to(device) 
	)

	inits = th.stack([x_init, y_init, z_init], dim=-2)
	fins = th.stack([x_fin, y_fin, z_fin], dim=-2)
	conditions = th.stack([inits, fins], dim=-1)

	state_x, state_y, state_z = make_state(
		x_init, y_init, z_init, x_fin, y_fin, z_fin
	)
	inp = make_inp(x_init, y_init, z_init, x_fin, y_fin, z_fin)

	with th.no_grad():
		start = perf_counter()
		latent = th.randn((batch_size, lheight * lwidth * zdim), device=device)
		if verbose:
			print(f"{'Random sampling time':<30}: {(perf_counter() - start):.4f} secs")
		
		start = perf_counter()
		c_pred = cv_model.decode(
			latent, conditions.repeat_interleave(batch_size, 0), 
			x_init.repeat_interleave(batch_size, 0), y_init.repeat_interleave(batch_size, 0), z_init.repeat_interleave(batch_size, 0), 
			x_fin.repeat_interleave(batch_size, 0), y_fin.repeat_interleave(batch_size, 0), z_fin.repeat_interleave(batch_size, 0)
		)
		if verbose:
			print(f"{'VQ decoder time':<30}: {(perf_counter() - start):.4f} secs")

		cx_pred, cy_pred, cz_pred = (
			c_pred[:, 0, ...].flatten(start_dim=-2), 
			c_pred[:, 1, ...].flatten(start_dim=-2), 
			c_pred[:, 2, ...].flatten(start_dim=-2)
		)
		c_pred_test = th.cat([cx_pred, cy_pred, cz_pred], dim=-1)

		inp_test = th.vstack([inp] * batch_size)
		cen_test = th.vstack([center] * batch_size)
		rad_test = th.vstack([radii] * batch_size)
		state_x_test = th.vstack([state_x]*batch_size)
		state_y_test = th.vstack([state_y]*batch_size)
		state_z_test = th.vstack([state_z]*batch_size)

		c_x_guess, c_y_guess, c_z_guess, lamda_x, lamda_y, lamda_z = model_sf(inp_test, th.cat([c_pred_test, cen_test, rad_test], dim=-1))
		
		c_x_pred = c_pred_test[:, 0 : multi_agent_learned.nvar*multi_agent_learned.num_agent] 
		c_y_pred = c_pred_test[:, multi_agent_learned.nvar*multi_agent_learned.num_agent : 2*multi_agent_learned.nvar*multi_agent_learned.num_agent]  
		c_z_pred = c_pred_test[:, 2*multi_agent_learned.nvar*multi_agent_learned.num_agent : 3*multi_agent_learned.nvar*multi_agent_learned.num_agent]


		x_ell, y_ell, z_ell = center.permute(1, 0)
		x_ell, y_ell, z_ell = (
			x_ell[..., None].cpu() * th.ones((batch_size, num * num_agent)),
			y_ell[..., None].cpu() * th.ones((batch_size, num * num_agent)),
			z_ell[..., None].cpu() * th.ones((batch_size, num * num_agent))
		)

		a_ell, b_ell, c_ell = radii.permute(1, 0)
		a_ell, b_ell, c_ell = (
			a_ell[..., None].cpu() * th.ones((batch_size, num * num_agent)),
			b_ell[..., None].cpu() * th.ones((batch_size, num * num_agent)),
			c_ell[..., None].cpu() * th.ones((batch_size, num * num_agent))
		)
		
		lamda_x_jnp = jnp.asarray( lamda_x.cpu().detach().numpy()  )
		lamda_y_jnp = jnp.asarray( lamda_y.cpu().detach().numpy()  )
		lamda_z_jnp = jnp.asarray( lamda_z.cpu().detach().numpy()  )
		
		c_x_pred_jnp = jnp.asarray( c_x_pred.cpu().detach().numpy()  )
		c_y_pred_jnp = jnp.asarray( c_y_pred.cpu().detach().numpy()  )
		c_z_pred_jnp = jnp.asarray( c_z_pred.cpu().detach().numpy()  )
		
		c_x_guess_jnp = jnp.asarray( c_x_guess.cpu().detach().numpy()  )
		c_y_guess_jnp = jnp.asarray( c_y_guess.cpu().detach().numpy()  )
		c_z_guess_jnp = jnp.asarray( c_z_guess.cpu().detach().numpy()  )

		c_x_guess = th.from_numpy(np.array(c_x_guess_jnp))
		c_y_guess = th.from_numpy(np.array(c_y_guess_jnp))
		c_z_guess = th.from_numpy(np.array(c_z_guess_jnp))
			
		state_x_test_jnp = jnp.asarray(state_x_test.cpu().detach().numpy())
		state_y_test_jnp = jnp.asarray(state_y_test.cpu().detach().numpy())
		state_z_test_jnp = jnp.asarray(state_z_test.cpu().detach().numpy())

		start = perf_counter()
		primal_sol_x, primal_sol_y, primal_sol_z, accumulated_res_primal, accumulated_res_fixed_point, primal_residual, dual_residual = jax.block_until_ready(multi_agent_learned(
			c_x_pred_jnp, c_y_pred_jnp, c_z_pred_jnp, c_x_guess_jnp, c_y_guess_jnp, c_z_guess_jnp, 
			lamda_x_jnp, lamda_y_jnp, lamda_z_jnp, state_x_test_jnp, state_y_test_jnp, state_z_test_jnp, 
			x_ell.cpu().numpy(), y_ell.cpu().numpy(), z_ell.cpu().numpy(), a_ell.cpu().numpy(), b_ell.cpu().numpy(), c_ell.cpu().numpy()
		))

		if verbose:
			print(f"{'Learned optimizer time':<30}: {(perf_counter() - start):.4f} secs")

		start = perf_counter()
		if comp_type == "base":
			primal_sol_x_base, primal_sol_y_base, primal_sol_z_base, primal_residual_base, dual_residual_base = multi_agent_base.custom_forward(
				state_x_test_jnp, state_y_test_jnp, state_z_test_jnp, 
				x_ell.cpu().numpy(), y_ell.cpu().numpy(), z_ell.cpu().numpy(), 
				a_ell.cpu().numpy(), b_ell.cpu().numpy(), c_ell.cpu().numpy()
			)
		elif comp_type == "sf_cpred":
			primal_sol_x_base, primal_sol_y_base, primal_sol_z_base, _, _, primal_residual_base, dual_residual_base = jax.block_until_ready(multi_agent_learned(
				c_x_pred_jnp, c_y_pred_jnp, c_z_pred_jnp, 
				c_x_pred_jnp, c_y_pred_jnp, c_z_pred_jnp, 
				jnp.zeros_like(lamda_x_jnp), jnp.zeros_like(lamda_y_jnp), jnp.zeros_like(lamda_z_jnp), 
				state_x_test_jnp, state_y_test_jnp, state_z_test_jnp, 
				x_ell.cpu().numpy(), y_ell.cpu().numpy(), z_ell.cpu().numpy(), 
				a_ell.cpu().numpy(), b_ell.cpu().numpy(), c_ell.cpu().numpy()
			))
		elif comp_type == "sf_zero":
			primal_sol_x_base, primal_sol_y_base, primal_sol_z_base, _, _, primal_residual_base, dual_residual_base = jax.block_until_ready(multi_agent_learned(
				c_x_pred_jnp, c_y_pred_jnp, c_z_pred_jnp, 
				jnp.zeros_like(c_x_pred_jnp), jnp.zeros_like(c_y_pred_jnp), jnp.zeros_like(c_z_pred_jnp), 
				jnp.zeros_like(lamda_x_jnp), jnp.zeros_like(lamda_y_jnp), jnp.zeros_like(lamda_z_jnp), 
				state_x_test_jnp, state_y_test_jnp, state_z_test_jnp, 
				x_ell.cpu().numpy(), y_ell.cpu().numpy(), z_ell.cpu().numpy(), 
				a_ell.cpu().numpy(), b_ell.cpu().numpy(), c_ell.cpu().numpy()
			))
		else:
			raise TypeError(f"{comp_type} is not recoginised")

		if verbose:
			print(f"{'Base/Comp optimizer time':<30}: {(perf_counter() - start):.4f} secs")

		try:
			best_idx = np.nanargmin(primal_residual[-1, :] + 0*dual_residual[-1, :])
			best_idx_base = np.nanargmin(primal_residual_base[-1, :] + 0*dual_residual_base[-1, :])
		except ValueError:
			raise 

		primal_sol_x_1 = primal_sol_x[best_idx]
		primal_sol_y_1 = primal_sol_y[best_idx]
		primal_sol_z_1 = primal_sol_z[best_idx]

		primal_sol_x_1 = primal_sol_x_1.reshape(multi_agent_learned.num_agent, multi_agent_learned.nvar)
		primal_sol_y_1 = primal_sol_y_1.reshape(multi_agent_learned.num_agent, multi_agent_learned.nvar)
		primal_sol_z_1 = primal_sol_z_1.reshape(multi_agent_learned.num_agent, multi_agent_learned.nvar)

		primal_sol_x_1_base = primal_sol_x_base[best_idx_base]
		primal_sol_y_1_base = primal_sol_y_base[best_idx_base]
		primal_sol_z_1_base = primal_sol_z_base[best_idx_base]

		primal_sol_x_1_base = primal_sol_x_1_base.reshape(multi_agent_learned.num_agent, multi_agent_learned.nvar)
		primal_sol_y_1_base = primal_sol_y_1_base.reshape(multi_agent_learned.num_agent, multi_agent_learned.nvar)
		primal_sol_z_1_base = primal_sol_z_1_base.reshape(multi_agent_learned.num_agent, multi_agent_learned.nvar)
		
		primal_sol_x_1_init = c_x_pred_jnp[best_idx]
		primal_sol_y_1_init = c_y_pred_jnp[best_idx]
		primal_sol_z_1_init = c_z_pred_jnp[best_idx]

		primal_sol_x_1_init = primal_sol_x_1_init.reshape(multi_agent_learned.num_agent, multi_agent_learned.nvar)
		primal_sol_y_1_init = primal_sol_y_1_init.reshape(multi_agent_learned.num_agent, multi_agent_learned.nvar)
		primal_sol_z_1_init = primal_sol_z_1_init.reshape(multi_agent_learned.num_agent, multi_agent_learned.nvar)

		x_traj = jnp.dot(multi_agent_learned.P, primal_sol_x_1.T).T
		y_traj = jnp.dot(multi_agent_learned.P, primal_sol_y_1.T).T
		z_traj = jnp.dot(multi_agent_learned.P, primal_sol_z_1.T).T

		x_traj_base = jnp.dot(multi_agent_learned.P, primal_sol_x_1_base.T).T
		y_traj_base = jnp.dot(multi_agent_learned.P, primal_sol_y_1_base.T).T
		z_traj_base = jnp.dot(multi_agent_learned.P, primal_sol_z_1_base.T).T
		
		x_traj_init = jnp.dot(multi_agent_learned.P, primal_sol_x_1_init.T).T
		y_traj_init = jnp.dot(multi_agent_learned.P, primal_sol_y_1_init.T).T
		z_traj_init = jnp.dot(multi_agent_learned.P, primal_sol_z_1_init.T).T

		x_ddot = jnp.dot(multi_agent_learned.Pddot, primal_sol_x[best_idx].reshape(num_agent, nvar).T).T
		y_ddot = jnp.dot(multi_agent_learned.Pddot, primal_sol_y[best_idx].reshape(num_agent, nvar).T).T
		z_ddot = jnp.dot(multi_agent_learned.Pddot, primal_sol_z[best_idx].reshape(num_agent, nvar).T).T

		x_ddot_base = jnp.dot(multi_agent_learned.Pddot, primal_sol_x_base[best_idx_base].reshape(num_agent, nvar).T).T
		y_ddot_base = jnp.dot(multi_agent_learned.Pddot, primal_sol_y_base[best_idx_base].reshape(num_agent, nvar).T).T
		z_ddot_base = jnp.dot(multi_agent_learned.Pddot, primal_sol_z_base[best_idx_base].reshape(num_agent, nvar).T).T

		ddot = jnp.stack([x_ddot, y_ddot, z_ddot], axis=-1)
		mean_smoothness = jnp.linalg.norm(ddot, axis=-1).sum(axis=-1).mean()

		ddot_base = jnp.stack([x_ddot_base, y_ddot_base, z_ddot_base], axis=-1)
		mean_smoothness_base = jnp.linalg.norm(ddot_base, axis=-1).sum(axis=-1).mean()

		return (
			primal_residual[:, best_idx], primal_residual_base[:, best_idx_base], 
			dual_residual[:, best_idx], dual_residual_base[:, best_idx_base]
		), (
			x_traj, y_traj, z_traj,
			x_traj_base, y_traj_base, z_traj_base,
			x_traj_init, y_traj_init, z_traj_init,
		), (mean_smoothness, mean_smoothness_base)
	

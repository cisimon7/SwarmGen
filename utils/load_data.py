import numpy as np
import torch as th
from torch.utils.data import Dataset

from pathlib import Path


class BoundedTrajectoryDataset(Dataset):
    def __init__(self, data, center, radii, P, num_agent=16, nvar=11):
        data = data.permute(0, 2, 1, 3)
        self.num_agents = num_agent
        self._filter_nans(
            data[..., 0], data[..., 1], data[..., 2], P
        )
        self.center = center
        self.radii = radii

    def _filter_nans(self, primal_sol_x, primal_sol_y, primal_sol_z, P):
        primal_sol = th.stack([primal_sol_x, primal_sol_y, primal_sol_z], dim=-1)
        
        mask = primal_sol.isnan().any(dim=(-1, -2, -3)).logical_not()
        primal_sol = primal_sol[mask]

        trajs = primal_sol.mT @ P.mT
        mask = ((primal_sol.mT @ P.cpu().mT).diff(dim=-1).mT.norm(dim=-1) >= 1e-3).sum(dim=-1).all(dim=(-1,)) 
        primal_sol = primal_sol[mask]

        self.traj_x, self.traj_y, self.traj_z = trajs.chunk(3, dim=-2)

    def __len__(self):
        return len(self.traj_x)

    def __getitem__(self, idx):
        return (
            self.traj_x[idx].squeeze(dim=-2), 
            self.traj_y[idx].squeeze(dim=-2), 
            self.traj_z[idx].squeeze(dim=-2)
        ), (self.center[idx], self.radii[idx])
    

def get_real_dataset_num_agents(P, num_agent=16, nvar=11):
    dir = Path(__file__).parent.parent / "resources" / "data" / "traj_data"
    if num_agent==32:
        data = np.load(dir / "trajs_30_32_agents.npz", allow_pickle=True)
    if num_agent==16:
        data = np.load(dir / "trajs_0_16_agents.npz", allow_pickle=True)

    primal_sol_x = data["primal_sol_x"]
    primal_sol_y = data["primal_sol_y"]
    primal_sol_z = data["primal_sol_z"]
    traj_data = np.stack([primal_sol_x, primal_sol_y, primal_sol_z], axis=-1).reshape(-1, num_agent, nvar, 3)
    traj_data = th.from_numpy(traj_data).permute(0, 2, 1, 3)
    dataset_traj = BoundedTrajectoryDataset(traj_data, data["center"], data["radii"], P, num_agent, nvar)
    return dataset_traj


class TrajDataset(Dataset):
    """Expert Trajectory Dataset."""
    def __init__(self, inp, state_x, state_y, state_z, c_pred, centers, radiis):
        
        self.inp = inp
        
        self.state_x = state_x
        self.state_y = state_y
        self.state_z = state_z
        
        self.centers = centers
        self.c_pred = c_pred
        self.radiis = radiis
         
    def __len__(self):
        return len(self.inp)    
            
    def __getitem__(self, idx):
        
        inp = self.inp[idx]
        state_x = self.state_x[idx]
        state_y = self.state_y[idx]
        state_z = self.state_z[idx]
        c_pred = self.c_pred[idx]
        center = self.centers[idx]
        radii = self.radiis[idx]
         
        return th.from_numpy(inp), th.from_numpy(state_x), th.from_numpy(state_y), th.from_numpy(state_z), th.from_numpy(c_pred), th.from_numpy(center), th.from_numpy(radii)
    

def load_multi_trajs_data(path, num_agent=16, nvar=11):
    dir = Path(__file__).parent.parent / "resources" / "data" / "multi_data"
    data = np.load(dir / path)
    condition_data = data["condition"].astype(np.float32)       # num_batch x 3 x num_agent x 6  num_batch, 18, num_agent
    primals_data = data["primals"].astype(np.float32)           # num_batch x 3 x num_agent x 11
    experts_data = data["experts"].astype(np.float32)           # num_batch x 3 x num_agent x 50
    centers = data["centers"].astype(np.float32) 
    radiis = data["radiis"].astype(np.float32)

    dataset_size = np.shape(primals_data)[0]

    inp = condition_data.reshape(dataset_size, 18, num_agent)
    c_pred = primals_data.reshape(dataset_size, 3*num_agent*nvar)

    x_init = condition_data[:, 0, :, 0]
    y_init = condition_data[:, 1, :, 0]
    z_init = condition_data[:, 2, :, 0]

    vx_init = condition_data[:, 0, :, 1]
    vy_init = condition_data[:, 1, :, 1]
    vz_init = condition_data[:, 2, :, 1]

    ax_init = condition_data[:, 0, :, 2]
    ay_init = condition_data[:, 1, :, 2]
    az_init = condition_data[:, 2, :, 2]

    x_fin = condition_data[:, 0, :, 3]
    y_fin = condition_data[:, 1, :, 3]
    z_fin = condition_data[:, 2, :, 3]

    vx_fin = condition_data[:, 0, :, 4]
    vy_fin = condition_data[:, 1, :, 4]
    vz_fin = condition_data[:, 2, :, 4]

    ax_fin = condition_data[:, 0, :, 5]
    ay_fin = condition_data[:, 1, :, 5]
    az_fin = condition_data[:, 2, :, 5] 

    inp_temp = np.hstack(( x_init, y_init, z_init, vx_init, vy_init, vz_init, ax_init, ay_init, az_init, x_fin, y_fin, z_fin, vx_fin, vy_fin, vz_fin, ax_fin, ay_fin, az_fin   ))
    inp = inp_temp.reshape(dataset_size, 18, num_agent)

    min_inp, max_inp = inp.min(), inp.max()
    inp_mean, inp_std = inp.mean(), inp.std()

    state_x = np.hstack(( x_init, vx_init, ax_init, x_fin, vx_fin, ax_fin  ))
    state_y = np.hstack(( y_init, vy_init, ay_init, y_fin, vy_fin, ay_fin  ))
    state_z = np.hstack(( z_init, vz_init, az_init, z_fin, vz_fin, az_fin  ))

    return TrajDataset(inp, state_x, state_y, state_z, c_pred, centers, radiis), min_inp, max_inp, inp_mean, inp_std
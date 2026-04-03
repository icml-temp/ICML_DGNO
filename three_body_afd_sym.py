import sys
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from functools import partial

# AFD Imports
from numpy import *
from numpy.matlib import repmat  # Fixed import
from numpy.fft import fft, ifft
from scipy.signal import hilbert


# ==========================================
# 0. SETTINGS & SEED
# ==========================================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"[Info] Random seed set to {seed}")


setup_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==========================================
# 1. Utilities
# ==========================================
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size(0)
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / (y_norms + 1e-8))
            else:
                return torch.sum(diff_norms / (y_norms + 1e-8))
        return diff_norms / (y_norms + 1e-8)

    def __call__(self, x, y):
        return self.rel(x, y)


def relative_l2_coef(pred, target, eps=1e-8):
    B = pred.shape[0]
    diff = (pred - target).reshape(B, -1)
    targ = target.reshape(B, -1)
    return (torch.norm(diff, dim=1) / (torch.norm(targ, dim=1) + eps)).mean()


# ==========================================
# 2. AFD Utilities
# ==========================================
def e_a(a, t):
    y = ((1 - absolute(a) ** 2) ** 0.5) / (1 - conjugate(a) * (e ** (1j * t)))
    return y


def weight(n, order):
    return ones((n, 1), 'complex')


def intg(f, g, W):
    if ndim(g) == 1: g = array([g])
    y = f.dot(g.T * W)
    if ndim(y) != 1: y = y[0, 0]
    return y / (shape(f)[1])


def FFT_AFD(s, max_level=50, M=20):
    if ndim(s) == 1: s = array([s])
    K = shape(s)[1]
    t = array([arange(0, 2 * pi, 2 * pi / K)])
    G = hilbert(s) if isreal(s).all() else s.copy()

    abs_a = array([arange(0, 1, 1.0 / M)])
    temp = zeros((1, size(abs_a)), 'complex')
    for k in arange(0, size(abs_a)): temp[0, k] = complex(abs_a[0, k])
    abs_a = temp.copy();
    del temp

    Base = zeros((size(abs_a), size(t)), 'complex')
    for k in arange(0, shape(Base)[0]): Base[k, :] = fft(e_a(abs_a[0, k], t), size(t))
    Weight = weight(K, 6)

    an = zeros((1, max_level + 1), 'complex')
    coef = zeros((1, max_level + 1), 'complex')
    coef[0, 0] = intg(G, ones((1, size(t))), Weight)

    for n in arange(1, size(an)):
        e_an = e_a(an[0, n - 1], t)
        G = (G - coef[0, n - 1] * e_an) * (1 - conjugate(an[0, n - 1]) * (e ** (1j * t))) / (
                    e ** (1j * t) - an[0, n - 1])
        S1 = ifft(repmat(fft(G * Weight.conj().T, size(t)), shape(Base)[0], 1) * Base, size(t), 1)
        max_loc = nonzero(absolute(S1) == absolute(S1).max())
        idx0 = max_loc[0][0] if len(max_loc[0]) > 0 else 0
        idx1 = max_loc[1][0] if len(max_loc[1]) > 0 else 0
        an[0, n] = abs_a[0, idx0] * e ** (1j * t[0, idx1])
        coef[0, n] = conjugate(e_a(an[0, n], t).dot(G.conj().T * Weight))[0, 0] / K
    return 1, an, coef, t


def component_AFD(an, coef, t):
    if ndim(an) == 1: an = array([an])
    if ndim(coef) == 1: coef = array([coef])
    if ndim(t) == 1: t = array([t])
    e_an = zeros((size(an), size(t)), 'complex')
    B_n = zeros((size(an), size(t)), 'complex')
    n = 0
    e_an[n, :] = e_a(an[0, n], t)
    B_n[n, :] = (sqrt(1 - absolute(an[0, 0]) ** 2) / (1 - conjugate(an[0, 0]) * e ** (t * 1j)))
    n = 1
    while n < size(an):
        e_an[n, :] = e_a(an[0, n], t)
        B_n[n, :] = (sqrt(1 - absolute(an[0, n]) ** 2) / (1 - conjugate(an[0, n]) * e ** (t * 1j))) * (
                    (e ** (1j * t) - an[0, n - 1]) / (sqrt(1 - absolute(an[0, n - 1]) ** 2))) * B_n[n - 1, :]
        n = n + 1
    return e_an, B_n


# ==========================================
# 3. DG Head (Restored!)
# ==========================================
class SymplecticGraphLayer(nn.Module):
    def __init__(self, hidden_dim, adj, dt=0.1):
        super(SymplecticGraphLayer, self).__init__()
        assert hidden_dim % 2 == 0
        self.d_half = hidden_dim // 2
        self.dt = dt
        self.register_buffer('adj', adj.float())
        self.net_V = nn.Sequential(nn.Linear(self.d_half, self.d_half), nn.GELU(), nn.Linear(self.d_half, self.d_half))
        self.net_T = nn.Sequential(nn.Linear(self.d_half, self.d_half), nn.GELU(), nn.Linear(self.d_half, self.d_half))
        self.msg_passing = nn.Linear(self.d_half, self.d_half)

    def forward(self, x):
        q, p = torch.chunk(x, 2, dim=-1)
        force_local = self.net_V(q)
        force_msg = self.msg_passing(force_local)
        force_global = torch.einsum('bkd, kj -> bjd', force_msg, self.adj)
        total_force = force_local + force_global
        p_new = p - self.dt * total_force
        velocity = self.net_T(p_new)
        q_new = q + self.dt * velocity
        return torch.cat([q_new, p_new], dim=-1)


class DGHeadAFD(nn.Module):
    def __init__(self, K, hidden_dim=64, num_layers=3, dt=0.1):
        super(DGHeadAFD, self).__init__()
        adj = torch.ones(K, K)  # Fully connected for AFD modes
        self.input_proj = nn.Linear(4, hidden_dim)  # [Re, Im, a_re, a_im]
        self.layers = nn.ModuleList([
            SymplecticGraphLayer(hidden_dim, adj=adj, dt=dt) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, 2)  # [Re_new, Im_new]

    def forward(self, nodes_in):
        h = self.input_proj(nodes_in)
        for layer in self.layers: h = layer(h)
        return self.output_proj(h)


# ==========================================
# 4. HNET (FNO + MLP)
# ==========================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels;
        self.out_channels = out_channels
        self.modes1 = modes1;
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))

    def compl_mul2d(self, a, b):
        op = torch.einsum
        return torch.stack([
            op("bixy,ioxy->boxy", a[..., 0], b[..., 0]) - op("bixy,ioxy->boxy", a[..., 1], b[..., 1]),
            op("bixy,ioxy->boxy", a[..., 1], b[..., 0]) + op("bixy,ioxy->boxy", a[..., 0], b[..., 1])
        ], dim=-1)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        x_ft = torch.view_as_real(x_ft)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, 2, device=x.device)
        m1, m2 = min(self.modes1, out_ft.size(2)), min(self.modes2, out_ft.size(3))
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])
        x = torch.fft.irfft2(torch.view_as_complex(out_ft), s=(x.size(-2), x.size(-1)), dim=(-2, -1), norm="ortho")
        return x


class PotentialFNO(nn.Module):
    def __init__(self, modes, width):
        super(PotentialFNO, self).__init__()
        self.fc0 = nn.Linear(3, width)  # (x, y, id)
        self.conv0 = SpectralConv2d(width, width, modes, modes)
        self.conv1 = SpectralConv2d(width, width, modes, modes)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, q):
        B = q.shape[0]
        grid = torch.linspace(0, 1, 3, device=q.device).view(1, 3, 1).expand(B, 3, 1)
        x = torch.cat([q, grid], dim=-1).unsqueeze(2)  # [B, 3, 1, 3]
        x = self.fc0(x).permute(0, 3, 1, 2)
        x1 = self.conv0(x);
        x2 = self.w0(x);
        x = F.gelu(x1 + x2)
        x1 = self.conv1(x);
        x2 = self.w1(x);
        x = F.gelu(x1 + x2)
        x = x.permute(0, 2, 3, 1)
        x = self.fc2(F.gelu(self.fc1(x)))  # [B, 3, 1, 1]
        return x.sum(dim=(1, 2, 3))


class KineticMLP(nn.Module):
    def __init__(self, width):
        super(KineticMLP, self).__init__()
        self.net = nn.Sequential(nn.Linear(6, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh(),
                                 nn.Linear(width, 1))

    def forward(self, p):
        return self.net(p.reshape(p.shape[0], -1)).squeeze(1)


class HamiltonianModel(nn.Module):
    def __init__(self, modes=4, width=64):
        super(HamiltonianModel, self).__init__()
        self.net_V = PotentialFNO(modes, width)
        self.net_T = KineticMLP(width)

    def forward(self, q, p): return self.net_V(q) + self.net_T(p)


# ==========================================
# 5. Configuration
# ==========================================
class Config:
    def __init__(self):
        self.system_type = '3body'
        self.noiseless_dataset_index = 'new_ivp_B'
        self.model_type = 'HNET_AFD_DG_FNO'

        self.T = 20
        self.dt = 0.1
        self.n_samples = 10000
        self.n_val_samples = 1000
        self.n_test_samples = 1000
        self.batch_size = 64
        self.n_epochs = 200
        self.lr = 1e-3

        # Model Params
        self.modes = 4
        self.width = 64

        # AFD / DG Params
        self.AFD_LEVELS = 5
        self.AFD_M = 20
        self.DG_HIDDEN = 64
        self.DG_LAYERS = 3

        # Loss Weights
        self.LAMBDA_AFD = 0.1  # Frequency domain regularization
        self.LAMBDA_DG = 0.01  # DG Head consistency loss

        self.data_dir = './data_3body_ivp'
        self.output_dir = './output_3body_hnet_afd_dg'


args = Config()
if not os.path.isdir(args.output_dir): os.makedirs(args.output_dir)


# ==========================================
# 6. Leapfrog Integrator
# ==========================================
def leapfrog_step(model, q, p, dt):
    q.requires_grad_(True)
    V = model.net_V(q)
    dV_dq = torch.autograd.grad(V.sum(), q, create_graph=True)[0]
    p_half = p - 0.5 * dt * dV_dq

    p_half.requires_grad_(True)
    T = model.net_T(p_half)
    dT_dp = torch.autograd.grad(T.sum(), p_half, create_graph=True)[0]
    q_next = q + dt * dT_dp

    q_next.requires_grad_(True)
    V_next = model.net_V(q_next)
    dV_dq_next = torch.autograd.grad(V_next.sum(), q_next, create_graph=True)[0]
    p_next = p_half - 0.5 * dt * dV_dq_next

    return q_next, p_next


# ==========================================
# 7. Training Logic
# ==========================================
def train_and_evaluate():
    print("=== HNET + AFD + DG Head (The Full Stack) ===")
    setup_seed(42)

    train_file = os.path.join(args.data_dir, f'train_data_3body_{args.noiseless_dataset_index}.npy')
    test_file = os.path.join(args.data_dir, f'test_data_3body_{args.noiseless_dataset_index}.npy')
    if not os.path.exists(train_file): raise FileNotFoundError("Run data generation first!")

    train_raw = torch.from_numpy(np.load(train_file)).float().permute(1, 0, 2)[:args.n_samples]
    test_raw = torch.from_numpy(np.load(test_file)).float().permute(1, 0, 2)[:args.n_test_samples]

    # Time Reversal
    train_rev = torch.flip(train_raw, dims=[1]).clone()
    train_rev[..., 0:6] = -train_rev[..., 0:6]
    train_combined = torch.cat([train_raw, train_rev], dim=0)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_combined),
                                               batch_size=args.batch_size, shuffle=True)

    # AFD Basis
    print("Building AFD Basis...")
    rep_traj = train_raw[0, :, 6].numpy().reshape(1, args.T)
    _, an_full, _, t_global = FFT_AFD(rep_traj, max_level=args.AFD_LEVELS - 1, M=args.AFD_M)
    _, B_n = component_AFD(np.zeros((1, args.AFD_LEVELS)), np.zeros((1, args.AFD_LEVELS)), t_global)

    B_real = torch.from_numpy(B_n.real).float().to(device)  # [Levels, T]
    B_imag = torch.from_numpy(B_n.imag).float().to(device)
    w_vec = torch.from_numpy(weight(args.T, 6).reshape(-1).real).float().to(device)  # [T]

    # 'a' parameters for DG Head input
    a_real = torch.from_numpy(an_full[:, :args.AFD_LEVELS].reshape(-1).real).float().to(device)
    a_imag = torch.from_numpy(an_full[:, :args.AFD_LEVELS].reshape(-1).imag).float().to(device)

    def project_to_afd(traj_q):
        # traj_q: [B, T, 3, 2] -> flatten spatial [B, 6, T] -> permute [B, 6, T]
        traj_flat = traj_q.reshape(traj_q.shape[0], args.T, -1).permute(0, 2, 1)  # [B, 6, T]

        # Apply weights to time dimension: [B, 6, T] * [1, 1, T]
        weighted_traj = traj_flat * w_vec.view(1, 1, -1)

        # Matmul: [B, 6, T] @ [T, Levels] -> [B, 6, Levels]
        Re = torch.matmul(weighted_traj, B_real.t())
        Im = -torch.matmul(weighted_traj, B_imag.t())

        return torch.stack([Re, Im], dim=-1)  # [B, 6, Levels, 2]

    # Models
    model = HamiltonianModel(modes=args.modes, width=args.width).to(device)
    dg_head = DGHeadAFD(K=args.AFD_LEVELS, hidden_dim=args.DG_HIDDEN, num_layers=args.DG_LAYERS, dt=args.dt).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(dg_head.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs)
    criterion = LpLoss()

    best_loss = float('inf')

    print("Start Training...")
    for ep in range(args.n_epochs):
        model.train();
        dg_head.train()
        t_loss = 0

        for batch in train_loader:
            gt = batch[0].to(device)  # [B, T, 12]

            p_curr = gt[:, 0, 0:6].reshape(-1, 3, 2)
            q_curr = gt[:, 0, 6:12].reshape(-1, 3, 2)

            pred_q = [q_curr]
            pred_p = [p_curr]

            # Rollout
            for t in range(args.T - 1):
                q_next, p_next = leapfrog_step(model, q_curr, p_curr, args.dt)
                pred_q.append(q_next)
                pred_p.append(p_next)
                q_curr, p_curr = q_next, p_next

            q_stack = torch.stack(pred_q, dim=1)
            p_stack = torch.stack(pred_p, dim=1)
            pred_traj = torch.cat([p_stack.reshape(gt.shape[0], args.T, 6), q_stack.reshape(gt.shape[0], args.T, 6)],
                                  dim=2)

            # 1. Main Trajectory Loss
            loss_traj = criterion(pred_traj, gt)

            # 2. DG Head Consistency Loss
            # Project generated trajectory to AFD
            c_pred = project_to_afd(q_stack)  # [B, 6, Levels, 2]

            # Aggregate to get "System State" for DG Head
            c_pred_agg = c_pred.mean(dim=1)  # [B, Levels, 2]

            # DG Head Input Construction
            nodes = torch.zeros(gt.shape[0], args.AFD_LEVELS, 4, device=device)
            nodes[..., 0] = c_pred_agg[..., 0]  # Re
            nodes[..., 1] = c_pred_agg[..., 1]  # Im
            nodes[..., 2] = a_real  # AFD param
            nodes[..., 3] = a_imag  # AFD param

            # DG Head Prediction (Self-Consistency)
            c_dg_out = dg_head(nodes)  # [B, Levels, 2]

            # Loss: Does the trajectory's spectral form match the DG Head's physics prediction?
            loss_dg = relative_l2_coef(c_dg_out, c_pred_agg)

            # 3. Spectral Smoothness (AFD Regularization)
            loss_afd = torch.mean(c_pred_agg ** 2)

            # Total Loss
            loss = loss_traj + args.LAMBDA_DG * loss_dg + args.LAMBDA_AFD * loss_afd

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()

        scheduler.step()
        t_loss /= len(train_loader)

        if ep % 5 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                idx = np.random.choice(len(test_raw), 200, replace=False)
                gt = test_raw[idx].to(device)

                p_curr = gt[:, 0, 0:6].reshape(-1, 3, 2)
                q_curr = gt[:, 0, 6:12].reshape(-1, 3, 2)
                preds = []
                for t in range(args.T):
                    if t == 0:
                        preds.append(torch.cat([p_curr.reshape(-1, 6), q_curr.reshape(-1, 6)], dim=1).unsqueeze(1))
                        continue

                    with torch.enable_grad():
                        q_next, p_next = leapfrog_step(model, q_curr, p_curr, args.dt)
                    preds.append(torch.cat([p_next.reshape(-1, 6), q_next.reshape(-1, 6)], dim=1).unsqueeze(1))
                    q_curr, p_curr = q_next, p_next

                full_pred = torch.cat(preds, dim=1)
                val_loss = criterion(full_pred, gt).item()

                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_hnet_afd_fno.pt'))

            print(f"Ep {ep} | Train {t_loss:.4f} | Val {val_loss:.4f} | Best {best_loss:.4f}")

    print("Generating Viz...")
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_hnet_afd_fno.pt')))

    idx = 0
    gt = test_raw[idx].to(device)
    p_curr = gt[0, 0:6].reshape(1, 3, 2)
    q_curr = gt[0, 6:12].reshape(1, 3, 2)

    preds_q, preds_p = [q_curr], [p_curr]
    for t in range(args.T - 1):
        with torch.enable_grad():
            q_next, p_next = leapfrog_step(model, q_curr, p_curr, args.dt)
        preds_q.append(q_next);
        preds_p.append(p_next)
        q_curr, p_curr = q_next, p_next

    pred_q = torch.cat(preds_q, dim=0).detach().cpu().numpy()
    gt_q = gt[:, 6:].reshape(-1, 3, 2).cpu().numpy()

    plt.figure(figsize=(6, 6))
    for i in range(3):
        plt.plot(gt_q[:, i, 0], gt_q[:, i, 1], '-', label=f'GT P{i}')
        plt.plot(pred_q[:, i, 0], pred_q[:, i, 1], '--', label=f'Pred P{i}')
        plt.plot(gt_q[0, i, 0], gt_q[0, i, 1], 'o')
    plt.legend();
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, 'hnet_afd_result.png'))
    print("Done.")


if __name__ == "__main__":
    train_and_evaluate()
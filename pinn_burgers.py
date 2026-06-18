"""
PINN for 1D viscous Burgers equation.

  u_t + u*u_x = nu * u_xx,   x in [-1, 1],  t in [0, 1]
  u(x, 0) = -sin(pi*x)
  u(-1, t) = u(1, t) = 0

Reference: Raissi et al. (2019) Physics-Informed Neural Networks.

Usage:
    python pinn_burgers.py
    python pinn_burgers.py --epochs 5000
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("PyTorch not found. Install with: pip install torch")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

class PINN(nn.Module):
    def __init__(self, layers=(2, 20, 20, 20, 20, 1)):
        super().__init__()
        net = []
        for i in range(len(layers) - 2):
            net += [nn.Linear(layers[i], layers[i+1]), nn.Tanh()]
        net += [nn.Linear(layers[-2], layers[-1])]
        self.net = nn.Sequential(*net)

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)


# ---------------------------------------------------------------------------
# Loss components
# ---------------------------------------------------------------------------

def pde_residual(model, x, t, nu=0.01 / np.pi):
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    u = model(x, t)

    u_t = torch.autograd.grad(u, t, torch.ones_like(u),
                               create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u),
                               create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                                create_graph=True)[0]
    return u_t + u * u_x - nu * u_xx


def sample_points(N_ic, N_bc, N_pde, device):
    # initial condition: t=0, x in [-1,1]
    x_ic = torch.FloatTensor(N_ic, 1).uniform_(-1, 1).to(device)
    t_ic = torch.zeros(N_ic, 1).to(device)
    u_ic = -torch.sin(np.pi * x_ic)

    # boundary: x=-1 and x=1, t in [0,1], u=0
    t_bc = torch.FloatTensor(N_bc, 1).uniform_(0, 1).to(device)
    x_bc_l = -torch.ones(N_bc, 1).to(device)
    x_bc_r =  torch.ones(N_bc, 1).to(device)

    # collocation points
    x_pde = torch.FloatTensor(N_pde, 1).uniform_(-1, 1).to(device)
    t_pde = torch.FloatTensor(N_pde, 1).uniform_(0, 1).to(device)

    return (x_ic, t_ic, u_ic,
            x_bc_l, x_bc_r, t_bc,
            x_pde, t_pde)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(epochs=3000, lr=1e-3, device_str='cpu'):
    device = torch.device(device_str)
    model  = PINN().to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.5)

    (x_ic, t_ic, u_ic,
     x_bc_l, x_bc_r, t_bc,
     x_pde, t_pde) = sample_points(200, 100, 2000, device)

    history = []
    for epoch in range(1, epochs + 1):
        opt.zero_grad()

        # IC loss
        u_pred_ic = model(x_ic, t_ic)
        loss_ic   = torch.mean((u_pred_ic - u_ic)**2)

        # BC loss
        loss_bc = (torch.mean(model(x_bc_l, t_bc)**2) +
                   torch.mean(model(x_bc_r, t_bc)**2))

        # PDE residual loss
        res     = pde_residual(model, x_pde.clone(), t_pde.clone())
        loss_pde = torch.mean(res**2)

        loss = loss_ic + loss_bc + loss_pde
        loss.backward()
        opt.step()
        scheduler.step()

        if epoch % 500 == 0 or epoch == 1:
            print(f"Epoch {epoch:5d}  loss={loss.item():.3e}  "
                  f"ic={loss_ic.item():.3e}  bc={loss_bc.item():.3e}  "
                  f"pde={loss_pde.item():.3e}")
            history.append(loss.item())

    return model, history


# ---------------------------------------------------------------------------
# Evaluation & plotting
# ---------------------------------------------------------------------------

def evaluate(model, device_str='cpu'):
    device = torch.device(device_str)
    Nx, Nt = 256, 100
    x = np.linspace(-1, 1, Nx)
    t = np.linspace(0, 1, Nt)
    X, T = np.meshgrid(x, t, indexing='ij')

    x_flat = torch.FloatTensor(X.flatten()[:, None]).to(device)
    t_flat = torch.FloatTensor(T.flatten()[:, None]).to(device)
    with torch.no_grad():
        u_pred = model(x_flat, t_flat).cpu().numpy().reshape(Nx, Nt)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # space-time plot
    im = axes[0].imshow(u_pred, extent=[-1, 1, 0, 1], origin='lower',
                         cmap='RdBu_r', aspect='auto')
    plt.colorbar(im, ax=axes[0], label='u(x,t)')
    axes[0].set_title('PINN prediction — 1D Burgers')
    axes[0].set_xlabel('x'); axes[0].set_ylabel('t')

    # slice at t=0.8
    t_idx = int(0.8 * Nt)
    axes[1].plot(x, u_pred[:, t_idx], 'b-', lw=2, label='PINN  t=0.8')
    axes[1].plot(x, -np.sin(np.pi * x), 'k--', lw=1, label='IC (t=0)')
    axes[1].set_xlabel('x'); axes[1].set_ylabel('u')
    axes[1].set_title('Profile at t = 0.8')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('pinn_burgers.png', dpi=150)
    plt.show()
    print("Saved pinn_burgers.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--gpu',    action='store_true')
    args = parser.parse_args()

    device = 'cuda' if (args.gpu and torch.cuda.is_available()) else 'cpu'
    print(f"Device: {device}")

    model, history = train(epochs=args.epochs, device_str=device)
    evaluate(model, device_str=device)

    plt.figure(figsize=(6, 3))
    plt.semilogy(range(0, args.epochs+1, 500)[1:], history)
    plt.xlabel('Epoch'); plt.ylabel('Total loss')
    plt.title('PINN training loss'); plt.tight_layout()
    plt.savefig('pinn_loss.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()

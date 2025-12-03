# concept_energy_line_demo.py
# ------------------------------------------------------------
# Minimal runnable demo of a relational energy model for a single concept: "line".
# It implements:
#   - Synthetic data generator for the "line" concept
#   - Relational energy network E_theta(x, a, w)
#   - SGLD samplers for x and a
#   - Inner-loop inference of concept codes (w_x, w_a) from few demos
#   - Outer training loop with contrastive + KL-like losses
#   - Qualitative visualization of identification (attention) and generation (entity positions)
#
# HOW TO RUN (Colab or local):
#   !pip install -r requirements.txt   # if needed
#   !python concept_energy_line_demo.py
#
# You can tweak HYPERPARAMS near the top to run faster/slower.
# ------------------------------------------------------------

import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import trange, tqdm
import matplotlib.pyplot as plt
import logging

from datasets.line import LineConceptDataset
from models.relational_energy import RelationalEnergy

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('main.log'),
        logging.StreamHandler()  
    ]
)


# --------------------- HYPERPARAMS ---------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Scene/event sizes
T = 2              # time steps (x0, x1)
N = 8              # number of entities
DX = 2             # features per entity: 2D position only

# Model
DW = 16            # dimension of concept code w
HIDDEN = 128

# Training
BATCH_SIZE = 128
DEMO_SHOTS = 5     # few-shot demos per concept instance
STEPS = 800        # training iterations (increase to 5000-10000 for better quality)
LR = 1e-3
K_SGLD = 10
ALPHA_X = 1e-2     # SGLD step for x
ALPHA_A = 5e-3     # SGLD step for a
LAMBDA_KL = 1.0

# Eval/visualization
EVAL_BATCH = 1     # visualize a single instance
ATTN_THRESHOLD = 0.5

# --------------------- SEEDING ---------------------
def seed_all(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all(SEED)

# --------------------- SGLD ---------------------
def sgld_step(var, grad, alpha):
    noise = torch.randn_like(var) * (alpha ** 0.5)
    return var + 0.5 * alpha * grad + noise

@torch.no_grad()
def sgld_optimize_x(E, x_init, a, w, steps=10, alpha=1e-2):
    x = x_init.clone().requires_grad_(True)
    for _ in range(steps):
        E_x = E(x, a, w).sum()
        grad, = torch.autograd.grad(E_x, x, create_graph=False)
        x = sgld_step(x, grad, alpha).detach().requires_grad_(True)
    return x.detach()

@torch.no_grad()
def sgld_optimize_a(E, x, a_init, w, steps=10, alpha=1e-2):
    a = a_init.clone().requires_grad_(True)
    for _ in range(steps):
        E_a = E(x, a, w).sum()
        grad, = torch.autograd.grad(E_a, a, create_graph=False)
        a = sgld_step(a, grad, alpha).detach().requires_grad_(True)
    return a.detach()

# --------------------- CONCEPT-CODE INFERENCE ---------------------
def infer_concept_codes(E, demos, DW, steps=10, lr=0.1):
    '''
    demos keys: x0, x1, a (each [B,T,N,DX], [B,T,N,DX], [B,N])
    Returns: w_x, w_a [B, DW]
    '''
    B = demos['x0'].shape[0]
    w_x = torch.randn(B, DW, device=demos['x0'].device, requires_grad=True)
    w_a = torch.randn(B, DW, device=demos['x0'].device, requires_grad=True)
    opt = torch.optim.SGD([w_x, w_a], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        Ex = E(demos['x1'], demos['a'], w_x)
        Ea = E(demos['x0'], demos['a'], w_a)
        loss = (Ex + Ea).mean()
        loss.backward()
        opt.step()
    return w_x.detach(), w_a.detach()



def collate(batch_list):
    # Stack single-sample dicts into batch tensors
    out = {}
    for k in batch_list[0]:
        out[k] = torch.stack([b[k] for b in batch_list], dim=0)
    return out

# --------------------- TRAINING LOOP ---------------------
def training_step(E, batch_demo, opt_theta, K=10, alpha_x=1e-2, alpha_a=5e-3, lam=1.0):
    # 1) infer w's from demos (stop-grad on theta for simplicity)
    w_x, w_a = infer_concept_codes(E, batch_demo, DW=DW, steps=K, lr=0.1)
    w_x, w_a = w_x.detach(), w_a.detach()  # detach after inference to stop grad on theta

    # 2) Use the same batch as both demo and training data
    # This preserves the concept codes without dilution
    x0, x1, a = batch_demo['x0'], batch_demo['x1'], batch_demo['a']
    a_init = torch.randn_like(a)
    x_tilde = sgld_optimize_x(E, x0, a, w_x, steps=K, alpha=alpha_x)
    a_tilde = sgld_optimize_a(E, x0, a_init, w_a, steps=K, alpha=alpha_a)

    # 3) losses
    Ex_pos = E(x1, a, w_x)
    Ex_neg = E(x_tilde, a, w_x)
    Ea_pos = E(x0, a, w_a)
    Ea_neg = E(x0, a_tilde, w_a)

    Lx = F.softplus(Ex_pos - Ex_neg).mean()
    La = F.softplus(Ea_pos - Ea_neg).mean()
    L_ml = Lx + La

    L_kl = (Ex_neg + Ea_neg).mean()

    loss = L_ml + lam * L_kl

    opt_theta.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(E.parameters(), 1.0)
    opt_theta.step()

    return {
        "loss": loss.item(),
        "L_ml": L_ml.item(),
        "L_kl": L_kl.item(),
        "Ex_pos": Ex_pos.mean().item(),
        "Ex_neg": Ex_neg.mean().item(),
        "Ea_pos": Ea_pos.mean().item(),
        "Ea_neg": Ea_neg.mean().item(),
    }

def make_demo_batch(dataset, shots=5, device="cpu"):
    # sample K shots and stack into a demo batch
    items = [dataset[random.randrange(len(dataset))] for _ in range(shots)]
    demo = collate(items)
    # Ensure x0 has T frames (we already store with 2 frames)
    for k in demo:
        demo[k] = demo[k].to(device)
    return demo

def main():
    logging.info(f"Using device: {DEVICE}")
    ds = LineConceptDataset(B=1, T=T, N=N, DX=2, num_samples=50_000, device=DEVICE)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, drop_last=True)

    E = RelationalEnergy(DX, DW, hidden=HIDDEN).to(DEVICE)
    opt = torch.optim.Adam(E.parameters(), lr=LR)

    pbar = trange(STEPS, desc="Training")
    running = {}
    dl_iter = iter(dl)
    
    for step in pbar:
        # Get a training batch
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            batch = next(dl_iter)

        for k in batch:
            batch[k] = batch[k].to(DEVICE)

        # Use the same batch for both demo (concept code inference) and training
        logs = training_step(E, batch, opt, K=K_SGLD, alpha_x=ALPHA_X, alpha_a=ALPHA_A, lam=LAMBDA_KL)
        for k,v in logs.items():
            running[k] = 0.97*running.get(k, v) + 0.03*v  # EMA for display
        pbar.set_postfix({k: f"{running[k]:.3f}" for k in ["loss","L_ml","L_kl"]})

    # --------------------- QUALITATIVE EVALUATION ---------------------
    E.eval()
    with torch.no_grad():
        demo = make_demo_batch(ds, shots=DEMO_SHOTS, device=DEVICE)
        w_x, w_a = infer_concept_codes(E, demo, DW=DW, steps=K_SGLD, lr=0.1)

        # Take a fresh eval instance
        eval_sample = collate([ds[0] for _ in range(EVAL_BATCH)])
        for k in eval_sample: eval_sample[k] = eval_sample[k].to(DEVICE)

    # Identification: infer a via SGLD with x fixed
    a_init = torch.randn_like(eval_sample["a"])
    a_tilde = sgld_optimize_a(E, eval_sample["x0"], a_init, w_a[:EVAL_BATCH], steps=K_SGLD, alpha=ALPHA_A)
    a_prob = torch.sigmoid(a_tilde)[0].detach().cpu().numpy()

    # Generation: infer x1 given a fixed attention (use the ground-truth attention as a demonstration)
    x_tilde = sgld_optimize_x(E, eval_sample["x0"], eval_sample["a"], w_x[:EVAL_BATCH], steps=K_SGLD, alpha=ALPHA_X)
    x0_np = eval_sample["x0"][0, 0, :, :2].detach().cpu().numpy()
    x1_np = x_tilde[0, 1, :, :2].detach().cpu().numpy()

    # --------------------- PLOTS ---------------------
    # 1) Attention bar plot (identification)
    plt.figure()
    plt.title("Identification: inferred attention probabilities per entity")
    plt.bar(np.arange(N), a_prob)
    plt.xlabel("Entity index")
    plt.ylabel("sigmoid(a)")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig("identification_attention.png")

    # 2) Generation scatter plot: initial vs generated final positions
    plt.figure()
    plt.title("Generation: initial (t=0) vs generated final (t=1) positions")
    plt.scatter(x0_np[:,0], x0_np[:,1], label="t=0")
    plt.scatter(x1_np[:,0], x1_np[:,1], label="generated t=1", marker="x")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("generation_positions.png")

    print("Saved figures: identification_attention.png, generation_positions.png")
    print("Done.")

if __name__ == "__main__":
    main()

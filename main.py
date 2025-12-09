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

# import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange
import matplotlib.pyplot as plt
import logging

from torch.utils.tensorboard import SummaryWriter
from datasets.line import LineConceptDataset
from models.relational_energy import RelationalEnergy

writer = SummaryWriter()

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
STEPS = 2000       # training iterations (increase to 5000-10000 for better quality)
LR = 3e-4
K_SGLD = 30        # increased for better sampling
ALPHA_X = 5e-3     # SGLD step for x
ALPHA_A = 1e-2     # SGLD step for a
LAMBDA_KL = 0.1    # reduced to focus on contrastive learning

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
    """SGLD step: moves DOWN the energy gradient to find low-energy states"""
    noise = torch.randn_like(var) * (alpha ** 0.5)
    return var - 0.5 * alpha * grad + noise  # NEGATIVE gradient to minimize energy

def sgld_optimize_x(E, x_init, a, w, steps=10, alpha=1e-2):
    x = x_init.clone().detach().requires_grad_(True)
    for _ in range(steps):
        E_x = E(x, a, w).sum()
        grad, = torch.autograd.grad(E_x, x, create_graph=False)
        x = sgld_step(x, grad, alpha).detach().requires_grad_(True)
    return x.detach()

def sgld_optimize_a(E, x, a_init, w, steps=10, alpha=1e-2):
    a = a_init.clone().detach().requires_grad_(True)
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
def training_step(E, batch_demo, opt_theta, step=None,K=10, alpha_x=1e-2, alpha_a=5e-3, lam=1.0):
    E.train()
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

    writer.add_scalar("Loss/train", loss, step)
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
    
    logging.info("Starting Training")
    for _ in pbar:
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            batch = next(dl_iter)

        for k in batch:
            batch[k] = batch[k].to(DEVICE)

        # print(batch['x0'].shape)
        # print(batch['x1'].shape)
        # print(batch['a'].shape)
        # print(batch)

        # Use the same batch for both demo (concept code inference) and training
        logs = training_step(E, batch, opt,step=_, K=K_SGLD, alpha_x=ALPHA_X, alpha_a=ALPHA_A, lam=LAMBDA_KL)
        for k,v in logs.items():
            running[k] = 0.97*running.get(k, v) + 0.03*v  # EMA for display
        pbar.set_postfix({k: f"{running[k]:.3f}" for k in ["loss","L_ml","L_kl"]})
    
    writer.flush()
    logging.info("Finished Training")

    # --------------------- QUALITATIVE EVALUATION ---------------------
    E.eval()
    
    # Infer concept codes (needs gradients for w_x, w_a optimization)
    demo = make_demo_batch(ds, shots=DEMO_SHOTS, device=DEVICE)
    w_x, w_a = infer_concept_codes(E, demo, DW=DW, steps=K_SGLD, lr=0.1)

    # Get eval sample
    eval_sample = collate([ds[0] for _ in range(EVAL_BATCH)])
    for k in eval_sample:
        eval_sample[k] = eval_sample[k].to(DEVICE)

    # Identification: infer a via SGLD with x fixed
    a_init = torch.randn_like(eval_sample["a"])
    a_tilde = sgld_optimize_a(E, eval_sample["x0"], a_init, w_a[:EVAL_BATCH], steps=K_SGLD, alpha=ALPHA_A)
    a_prob = torch.sigmoid(a_tilde)[0].detach().cpu().numpy()

    # Generation: infer x1 given a fixed attention (use the ground-truth attention as a demonstration)
    x_tilde = sgld_optimize_x(E, eval_sample["x0"], eval_sample["a"], w_x[:EVAL_BATCH], steps=K_SGLD, alpha=ALPHA_X)
    x0_np = eval_sample["x0"][0, 0, :, :2].detach().cpu().numpy()
    x1_np = x_tilde[0, 1, :, :2].detach().cpu().numpy()

    # --------------------- PLOTS ---------------------
    # Get ground truth attention for visualization
    gt_a = eval_sample["a"][0].detach().cpu().numpy()
    attended_idx = np.where(gt_a > 1.0)[0]  # entities with high attention
    
    # 1) Attention comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.bar(np.arange(N), gt_a, alpha=0.7, label="Ground Truth")
    ax1.set_title("Ground Truth Attention")
    ax1.set_xlabel("Entity index")
    ax1.set_ylabel("Attention value")
    ax1.set_ylim(-0.5, 3.5)
    ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
    
    ax2.bar(np.arange(N), a_prob, alpha=0.7, color='orange', label="Inferred")
    ax2.set_title("Inferred Attention (Identification)")
    ax2.set_xlabel("Entity index")
    ax2.set_ylabel("sigmoid(a)")
    ax2.set_ylim(0, 1.0)
    ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("identification_attention.png")

    # 2) Generation visualization with lines
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Initial positions (t=0)
    ax1.scatter(x0_np[:,0], x0_np[:,1], c='gray', s=100, alpha=0.5, label="all entities")
    ax1.scatter(x0_np[attended_idx,0], x0_np[attended_idx,1], c='blue', s=150, 
                marker='o', label="attended entities", edgecolors='black', linewidths=2)
    ax1.set_title("Initial State (t=0)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.axis("equal")
    ax1.grid(True, alpha=0.3)
    
    # Right: Generated final positions (t=1) with line fit
    ax2.scatter(x1_np[:,0], x1_np[:,1], c='gray', s=100, alpha=0.5, label="all entities")
    
    # Get attended entities from inferred attention (>0.5 threshold)
    inferred_attended = np.where(a_prob > 0.5)[0]
    if len(inferred_attended) >= 2:
        ax2.scatter(x1_np[inferred_attended,0], x1_np[inferred_attended,1], 
                   c='red', s=150, marker='x', label="inferred line entities", linewidths=3)
        
        # Fit and draw line through attended points
        attended_points = x1_np[inferred_attended]
        # Sort by x coordinate for cleaner line visualization
        sorted_idx = np.argsort(attended_points[:, 0])
        sorted_points = attended_points[sorted_idx]
        ax2.plot(sorted_points[:, 0], sorted_points[:, 1], 'r--', 
                linewidth=2, alpha=0.6, label="fitted line")
    
    ax2.set_title("Generated State (t=1) - Line Formation")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend()
    ax2.axis("equal")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("generation_positions.png")

    logging.info("Saved figures: identification_attention.png, generation_positions.png")

if __name__ == "__main__":
    main()

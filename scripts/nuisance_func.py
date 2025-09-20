import torch
from BSpline import BSpline
from torch import Tensor
from torch import nn
from tqdm import tqdm

class BSplineTwoSample(nn.Module):
    def __init__(self, bspline_args, device):
        super().__init__()
        self.bspline = BSpline(**bspline_args)
        self.bspline = self.bspline.to(device)
        pass
    
    def inv_sqrt_matrix(self, M: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        # 1) eigen-decompose
        if M.device.type == "mps":
            M_cpu = M.detach().to("cpu", dtype=torch.float32)
            w, Q = torch.linalg.eigh(M_cpu)
            w = w.to(M.device)
            Q = Q.to(M.device)
        else:
            w, Q = torch.linalg.eigh(M)                # w: (d,), Q: (d,d)
        # 2) take inverse square-root of eigenvalues
        w_inv_sqrt = (w.clamp(min=eps) ** -0.5)     # (d,)
        D_inv_sqrt = torch.diag(w_inv_sqrt)        # (d,d)
        # 3) reconstruct
        return Q @ D_inv_sqrt @ Q.T                # (d,d)

    def solve_beta_star(self, A: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """
        Solves
            max_{β}  c^T β / sqrt(β^T A β)
            s.t.    d^T β = 0,
        returning the optimizer β* of unit A‐norm (i.e. β*^T A β* = 1).

        A: (d, d) symmetric positive‐definite
        c, d: (d,)
        """
        REGULARIZATION = 0.01
        # Solve B u = a
        alpha = REGULARIZATION * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)  # Regularization term

        # Solve A v = c  and  A w = d  (both on the same device & dtype)
        v = torch.linalg.solve(A+alpha, c)  # shape (d,)
        w = torch.linalg.solve(A+alpha, d)  # shape (d,)

        # Compute μ = (d^T A^{-1} c) / (d^T A^{-1} d) = (d·v)/(d·w)
        mu = torch.dot(d, v) / torch.dot(d, w)

        # Un‐normalized solution β₀ = A⁻¹(c – μ d) = v – μ w
        beta0 = v - mu * w            # shape (d,)

        # Normalize so that β*^T A β* = 1
        # First compute norm² = β₀ᵀ A β₀
        norm_sq = torch.dot(beta0, (A+alpha) @ beta0)
        norm = torch.sqrt(norm_sq)

        beta_star = beta0 / norm
        return beta_star

    def compute_beta_hat(
        self,
        z_u_list,   # human texts: list[torch.Tensor], each of shape (1, Li)
        z_v_list,   # LLM  texts: list[torch.Tensor], each of shape (1, Lj)
        constraint,
    ) -> torch.Tensor:
        device = z_u_list[0].device
        d = self.bspline.n_bases
        if self.bspline.add_intercept:
            d = d + 1

        # 1) Gather lengths and flatten all the z's at once
        u_lengths = [z.shape[-1] for z in z_u_list]
        v_lengths = [z.shape[-1] for z in z_v_list]

        # stack all the tokens into one long 1D tensor
        all_u = torch.cat([z.squeeze(0).clamp_min(self.bspline.start) for z in z_u_list], dim=0).to(device)
        all_v = torch.cat([z.squeeze(0).clamp_min(self.bspline.start) for z in z_v_list], dim=0).to(device)

        # 2) Compute B‑spline basis in one go
        all_u_feats = self.bspline(all_u)  # shape = (sum(u_lengths), d)
        all_v_feats = self.bspline(all_v)  # shape = (sum(v_lengths), d)

        # 3) Split back into per‑sequence tensors
        #    torch.split is implemented in C, so it's very cheap
        u_feats = list(torch.split(all_u_feats, u_lengths, dim=0))
        v_feats = list(torch.split(all_v_feats, v_lengths, dim=0))

        # 4) Compute u_means and v_means
        #    stacked means, shape = (n_u, d) and (n_v, d)
        u_means = torch.stack([f.mean(dim=0) for f in u_feats], dim=0)
        v_means = torch.stack([f.mean(dim=0) for f in v_feats], dim=0)

        # 5) Build delta
        delta = v_means.sum(dim=0) - u_means.sum(dim=0)     # (d,)

        # --- 3) Covariances Σ_u, Σ_v over ALL tokens in each sequence ---
        Sigma_u = torch.zeros((d, d), device=device)  # (d, d)
        for i, Fu in enumerate(u_feats):
            Fu_c = Fu - Fu.mean(dim=0, keepdim=True)   
            Sigma_u += ((Fu_c.T @ Fu_c) / (Fu_c.shape[0] - 1)) / Fu.shape[0] 
        Sigma_v = torch.zeros((d, d), device=device)
        for i, Fv in enumerate(v_feats):
            Fv_c = Fv - Fv.mean(dim=0, keepdim=True)   
            Sigma_v += ((Fv_c.T @ Fv_c) / (Fv_c.shape[0] - 1)) / Fv.shape[0]  
        Sigma = Sigma_u + Sigma_v                           # (d, d)

        # --- 4) Closed-form beta = Σ^{-1} δ, then normalize ---
        if constraint:
            beta_hat = self.solve_beta_star(Sigma, delta, u_means.sum(dim=0))
        else:
            Sigma = self.inv_sqrt_matrix(Sigma)
            beta_tilde = Sigma @ delta       # (d,)
            beta_hat   = beta_tilde / beta_tilde.norm(p=2)
        return beta_hat

    def get_zij(self, token_list, model, args):
        model.eval()

        n_samples = len(token_list)
        z_list = []
        for idx in tqdm(range(n_samples)):
            tokenized = token_list[idx]
            labels = tokenized.input_ids[:, 1:]
            with torch.no_grad():
                logits_score = model(**tokenized).logits[:, :-1]
            labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
            z_j_b = torch.log_softmax(logits_score, dim=-1)
            z_j = z_j_b.gather(dim=-1, index=labels).squeeze(-1)
            z_list.append(z_j)

        return z_list

    def fit(self, human_token_list, machine_token_list, model, args, constraint=False):
        print("Learning witness function...")
        print("Fetch log-likelihood of human texts...")
        z_ij_u = self.get_zij(human_token_list, model, args)
        print("Fetch log-likelihood of LLM texts...")
        z_ij_v = self.get_zij(machine_token_list, model, args)
        beta_hat = self.compute_beta_hat(z_ij_u, z_ij_v, constraint)
        self.beta_hat = beta_hat
        print("beta_hat:", torch.round(beta_hat, decimals=3))

    def forward(self, input: Tensor):
        input_shape = input.shape
        device = input.device
        flat = input.clamp_min(self.bspline.start).reshape(-1).to(device)
        w_value = self.bspline(flat) @ self.beta_hat
        w_value = w_value.reshape(input_shape)
        return w_value

def get_ci_list(text_list, tokenizer, model, w_fun, args):
    model.eval()

    n_samples = len(text_list)
    c_list = []
    for idx in tqdm(range(n_samples)):
        original_text = text_list[idx]
        tokenized = tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = model(**tokenized).logits[:, :-1]
        labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
        z_j_b = w_fun(torch.log_softmax(logits_score, dim=-1))
        probs_ref = torch.softmax(logits_score, dim=-1)
        mean_ref = (probs_ref * z_j_b).sum(dim=-1)
        z_j = z_j_b.gather(dim=-1, index=labels).squeeze(-1)
        
        ci = (z_j.mean(dim=-1) - mean_ref.mean(dim=-1))[0]
        c_list.append(ci)
    return c_list

class ShiftLearner(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def fit(self, data, tokenizer, model, w_func, args):
        print("Learning shift...")
        ci_hat_list = get_ci_list(data['original'], tokenizer, model, w_func, args)
        c_hat = torch.mean(torch.tensor(ci_hat_list))
        self.c_hat = c_hat
        print("c_hat:", torch.round(c_hat, decimals=3))

import torch
from torch import nn
from tqdm import tqdm
from BSpline import BSpline


def optimal_beta(a: torch.Tensor, B: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
    REGULARIZATION = 0.01
    # Solve B u = a
    alpha = REGULARIZATION * torch.eye(B.shape[0], device=B.device, dtype=B.dtype)  # Regularization term
    u = torch.linalg.solve(B+alpha, a)
    if c is None:
        w = u
    else:
        # Solve B v = c
        v = torch.linalg.solve(B+alpha, c)
        mu = torch.dot(c, u) / torch.dot(c, v)
        w = u - mu * v
    # Normalize to satisfy w^T B w = 1
    norm = torch.sqrt(torch.dot(w, B @ w))
    return w / norm

def get_bias_vector_mc(token_list, model, bspline: BSpline, args, compute_cov: bool = False, sample_size: int = 5000):
    """
    Monte Carol based method
    sample_size: sampling size 
    """
    model.eval()
    device = args.device
    n_bases = bspline.n_bases
    if bspline.add_intercept:
        n_bases += 1
    bias_list = []
    cov_list = []

    for tokens in tqdm(token_list):
        input_ids = tokens.input_ids[0]
        with torch.no_grad():
            logits = model(**tokens).logits[0, :-1]  # (seq_len, vocab_size)
        seq_len, vocab_size = logits.shape

        probs = torch.softmax(logits, dim=-1)     # (seq_len, vocab)
        logp = torch.log_softmax(logits, dim=-1)  # (seq_len, vocab)

        labels = input_ids[1:]  # (seq_len,)
        log_ll = logp[torch.arange(seq_len), labels]  # (seq_len,)
        w_j = bspline.predict(log_ll.clamp_min(bspline.start).reshape(-1)).to(device)
        w_j = w_j.reshape(seq_len, n_bases)

        sampled_indices = torch.multinomial(probs, num_samples=sample_size, replacement=True)
        
        sampled_logp = logp.gather(1, sampled_indices)
        
        flat_sampled_logp = sampled_logp.clamp_min(bspline.start).reshape(-1)
        flat_basis_samples = bspline.predict(flat_sampled_logp).to(device)
        basis_samples = flat_basis_samples.reshape(seq_len, sample_size, n_bases)
        
        mean_ref = basis_samples.mean(dim=1)  # (seq_len, n_bases)

        bias_sample = (w_j - mean_ref).sum(dim=0)  # (n_bases,)
        bias_list.append(bias_sample)

        if compute_cov:
            cov_sample = torch.zeros(n_bases, n_bases, device=device)
            
            for t in range(seq_len):
                phi_samples = basis_samples[t]
                
                sample_mean = phi_samples.mean(dim=0)
                
                centered_samples = phi_samples - sample_mean
                cov_t = (centered_samples.t() @ centered_samples) / (sample_size - 1)
                
                if bspline.add_intercept:
                    cov_t[0, :] = 0.0
                    cov_t[:, 0] = 0.0
                    
                cov_sample += cov_t
                
            cov_list.append(cov_sample)

    bias_vector = torch.stack(bias_list, dim=0).mean(dim=0)  # (n_bases,)
    if compute_cov:
        cov_matrix = torch.stack(cov_list, dim=0).mean(dim=0)  # (n_bases, n_bases)
        return bias_vector, cov_matrix
    return bias_vector

def get_bias_vector(token_list, model, bspline: BSpline, args, compute_cov: bool = False, speedup_rate = 1):
    """
    For each text in text_list, compute the bias vector
    (mean difference between sampled basis and expected basis)
    and, if requested, the covariance matrix of basis differences.
    Returns bias (n_bases,) and optionally cov (n_bases, n_bases).
    """
    model.eval()
    device = args.device
    n_bases = bspline.n_bases
    if bspline.add_intercept:
        n_bases += 1
    bias_list = []
    cov_list = []

    for tokens in tqdm(token_list):
        input_ids = tokens.input_ids[0]
        with torch.no_grad():
            logits = model(**tokens).logits[0, :-1]  # (seq_len, vocab_size)
        seq_len, vocab_size = logits.shape

        probs = torch.softmax(logits, dim=-1)     # (seq_len, vocab)
        vocab_size = int(vocab_size / speedup_rate)
        probs, _ = torch.topk(probs, k=vocab_size, dim=-1)
        probs = probs / probs.sum(dim=-1, keepdim=True)

        logp = torch.log_softmax(logits, dim=-1)  # (seq_len, vocab)

        # Basis at the actual next-token labels
        labels = input_ids[1:]                                     # (seq_len,)
        # Index basis per position
        log_ll = logp[torch.arange(seq_len), labels]              # (seq_len, n_bases)
        w_j = bspline.predict(log_ll.clamp_min(bspline.start).reshape(-1)).to(device)
        w_j = w_j.reshape(seq_len, n_bases)

        # Expected basis at the next-token
        logp, _ = torch.topk(logp, k=vocab_size, dim=-1)
        # Compute basis for all log-probabilities
        flat_logp = logp.clamp_min(bspline.start).reshape(-1)
        flat_basis = bspline.predict(flat_logp).to(device)        # (seq_len*vocab, n_bases)
        basis = flat_basis.reshape(seq_len, vocab_size, n_bases)    # (seq_len, vocab, n_bases)
        # Expected basis per position: E[basis] = sum_k p_{ik} * basis_{ik}
        mean_ref = (probs.unsqueeze(-1) * basis).sum(dim=1)       # (seq_len, n_bases)

        # Bias: average difference over positions
        bias_sample = (mean_ref - w_j).sum(dim=0)                 # (n_bases,)
        bias_list.append(bias_sample)

        if compute_cov:
            ###### Naive version ######
            cov_sample = torch.zeros(n_bases, n_bases, device=device)
            for t in range(seq_len):
                p_t = probs[t]  # (vocab,)
                phi_t = basis[t]  # (vocab, n_bases)
                # E_b[phi_t]: (n_bases,)
                Ex_t = (p_t.unsqueeze(1) * phi_t).sum(dim=0)
                # E_b[phi_t phi_t^T]: (n_bases, n_bases)
                ExxT_t = phi_t.t() @ (p_t.unsqueeze(1) * phi_t)
                cov_t = ExxT_t - Ex_t.unsqueeze(1) @ Ex_t.unsqueeze(0)
                cov_sample += cov_t
            ###### Faster version ######
            # wb = probs.unsqueeze(-1) * basis         # (T, V, K)
            # # 2) compute E[φ φᵀ] summed over t,b:
            # #    for each t: basis[t].T @ wb[t]   →  (K, K)
            # #    sum over t with a single bmm + sum:
            # ExxT = torch.sum(torch.bmm(basis.transpose(1,2), wb), dim=0)                                         # (K, K)
            # Ex_t      = wb.sum(dim=1)                                # (T, K)
            # sum_outer = torch.einsum('tk,tl->kl', Ex_t, Ex_t)        # (K, K)
            # cov_sample = ExxT - sum_outer
            
            cov_list.append(cov_sample)

    bias_vector = torch.stack(bias_list, dim=0).mean(dim=0)        # (n_bases,)
    if compute_cov:
        cov_matrix = torch.stack(cov_list, dim=0).mean(dim=0)      # (n_bases, n_bases)
        return bias_vector, cov_matrix
    return bias_vector


class BSplineTheory(nn.Module):
    def __init__(self, bspline_args, machine_text: bool = False):
        super().__init__()
        self.bspline = BSpline(**bspline_args)
        self.machine_text = machine_text
        self.beta_hat = None

    def fit(self, human_token_list, machine_token_list, model, args):
        device = args.device
        print("Learning w function...")

        print("Fetching bias and covariance for human texts...")
        bias_a, cov_B = get_bias_vector(human_token_list, model, self.bspline, args, compute_cov=True)
        print("Computing beta_hat...")
        # print("bias_a:", torch.round(bias_a, decimals=2))
        # print("cov_B:", torch.round(cov_B, decimals=2))

        if self.machine_text:
            print("Fetching bias for machine-generated texts...")
            bias_c = get_bias_vector(machine_token_list, model, self.bspline, args, compute_cov=False)
            # print("bias_c:", torch.round(bias_c, decimals=2))
        else:
            bias_c = None

        self.beta_hat = optimal_beta(
            bias_a.to(device), cov_B.to(device),
            bias_c.to(device) if bias_c is not None else None
        )
        print("beta_hat:", torch.round(self.beta_hat, decimals=3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        flat = x.clamp_min(self.bspline.start).reshape(-1)
        basis = self.bspline.predict(flat).to(device)   # (flat_len, n_bases)
        w_flat = basis @ self.beta_hat.to(device)      # (flat_len,)
        return w_flat.reshape(x.shape)

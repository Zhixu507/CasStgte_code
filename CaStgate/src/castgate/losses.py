import torch
from torch.nn import functional as F


def reliability_target(logits, labels, beta=1.0, temperature=0.5, component="joint"):
    """logits: [batch, aligned views including original, branches, classes].

    Targets are detached. Correctness is unweighted NLL, not class-reweighted CE.
    Generalized JS = average KL(view || view-mean), evaluated in probability space.
    """
    if temperature <= 0 or beta < 0:
        raise ValueError("temperature must be positive and beta nonnegative")
    with torch.no_grad():
        logp = logits.detach().float().log_softmax(-1)
        p = logp.exp()
        selected = labels[:, None, None, None].expand(*logp.shape[:-1], 1)
        nll = -logp.gather(-1, selected).squeeze(-1).mean(1)
        mean_p = p.mean(1, keepdim=True)
        js = (p * (logp - mean_p.clamp_min(1e-8).log())).sum(-1).mean(1).clamp_min(0)
        if component == "accuracy":
            risk = nll
        elif component == "stability":
            risk = js
        elif component == "joint":
            risk = nll + beta * js
        else:
            raise ValueError(f"Unknown reliability component: {component}")
        return (-risk / temperature).softmax(-1), {"nll": nll, "js": js}


def training_loss(outputs, labels, batch_size, views, config, warmup=False, teacher_logits=None):
    logits = outputs["branch_logits"].reshape(batch_size, views, 2, -1)
    probs = outputs["probs"].reshape(batch_size, views, -1)
    weights = outputs["weights"].reshape(batch_size, views, 2)
    # Supervise fused predictions on all views, equal for every fusion baseline.
    repeated = labels[:, None].expand(-1, views).reshape(-1)
    fused_ce = F.nll_loss(probs.reshape(-1, probs.shape[-1]).clamp_min(1e-8).log(), repeated)
    if config["mode"] in ("cnn", "transformer"):
        return fused_ce, {"loss": float(fused_ce.detach()), "fused_ce": float(fused_ce.detach()),
                          "branch_ce": 0., "gate_kl": 0.}
    branch_ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               labels[:, None, None].expand(-1, views, 2).reshape(-1))
    gate_loss = fused_ce.new_zeros(())
    if config["mode"] == "reliability_gate" and not warmup:
        target_logits = logits if teacher_logits is None else teacher_logits
        q, _ = reliability_target(target_logits, labels, beta=config["stability_beta"],
                                  temperature=config["temperature"], component=config["reliability"])
        if config.get("shuffle_gate_target"):
            # A batch permutation keeps the marginal target distribution and drops image alignment.
            if q.shape[0] < 2:
                raise ValueError("Shuffled reliability targets need at least two images in the batch")
            q = q[torch.randperm(q.shape[0], device=q.device)]
        # Supervise the ORIGINAL view's gate: this is the inference setting.
        gate_loss = F.kl_div(weights[:, 0].clamp_min(1e-8).log(), q, reduction="batchmean")
    loss = fused_ce + config["lambda_branch"] * branch_ce + config["lambda_gate"] * gate_loss
    return loss, {"loss": float(loss.detach()), "fused_ce": float(fused_ce.detach()),
                  "branch_ce": float(branch_ce.detach()), "gate_kl": float(gate_loss.detach())}

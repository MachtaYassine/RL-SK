def check_optimizer_gradients(optimizer, logger):
    missing = []
    for group in optimizer.param_groups:
        for idx, param in enumerate(group['params']):
            if param.requires_grad:
                if param.grad is None:
                    missing.append(idx)
                    logger.warning(f"Parameter {idx} in optimizer group has no gradient.")
                # else:
                #     logger.debug(f"Parameter {idx} grad norm: {param.grad.norm().item()}")
    return missing

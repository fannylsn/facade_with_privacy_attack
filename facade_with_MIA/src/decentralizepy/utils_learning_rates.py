def get_lr_step_5_8(ratio, lr):
    """Get learning rate for schedule with step ratios 0.5 and 0.8"""
    if 0.5 < ratio and ratio < 0.8:
        lr = lr * 0.5
    elif ratio >= 0.8:
        lr = lr * 0.1
    return lr


def get_lr_step_7_9(ratio, lr):
    """Get learning rate for schedule with step ratios 0.7 and 0.9"""
    if 0.7 < ratio and ratio < 0.9:
        lr = lr * 0.5
    elif ratio >= 0.9:
        lr = lr * 0.1
    return lr

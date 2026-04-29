"""Redirect: unifiedefficientloader.uel.torch → comfy_aimdo.torch

Adds backward-compat alias uel_to_tensor for aimdo_to_tensor.
"""
import sys
import comfy_aimdo.torch

# Backward-compat alias
if hasattr(comfy_aimdo.torch, "aimdo_to_tensor") and not hasattr(comfy_aimdo.torch, "uel_to_tensor"):
    comfy_aimdo.torch.uel_to_tensor = comfy_aimdo.torch.aimdo_to_tensor

sys.modules[__name__] = comfy_aimdo.torch

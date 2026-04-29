"""Redirect: unifiedefficientloader.uel.model_vbar → comfy_aimdo.model_vbar"""
import sys
import comfy_aimdo.model_vbar
sys.modules[__name__] = comfy_aimdo.model_vbar

"""Redirect: unifiedefficientloader.uel.host_buffer → comfy_aimdo.host_buffer"""
import sys
import comfy_aimdo.host_buffer
sys.modules[__name__] = comfy_aimdo.host_buffer

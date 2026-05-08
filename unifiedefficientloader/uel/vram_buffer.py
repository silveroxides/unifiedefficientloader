"""Redirect: unifiedefficientloader.uel.vram_buffer → comfy_aimdo.vram_buffer"""
import sys
import comfy_aimdo.vram_buffer
sys.modules[__name__] = comfy_aimdo.vram_buffer

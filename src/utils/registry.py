#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on 12/02/2024 16:25:42

@author: prdvanderwal
"""


def register_model(func):
    """
    Fallback wrapper in case timm isn't installed
    """
    return func

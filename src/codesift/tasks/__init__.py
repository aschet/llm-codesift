# SPDX-FileCopyrightText: 2026 Thomas Ascher <thomas.ascher@gmx.at>
#
# SPDX-License-Identifier: MIT
"""The tasks the screen puts to a model, one module per kind.

Small on purpose: this filters models that cannot do the work, it does not rank
the ones that can.
"""
from . import codegen, edit, format, toolcall, trace

TASKS = (codegen.TASKS + edit.TASKS + format.TASKS
         + toolcall.TASKS + trace.TASKS)

__all__ = ["TASKS"]

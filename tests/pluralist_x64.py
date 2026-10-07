"""Scope David Hyland's float64 setting to the pluralist tests only.

src/pluralist/model.py (copied verbatim) enables jax_enable_x64 globally on
import, as the original repository does. Inside one pytest process that would
leak into the float32 suites (it NaNs test_tier1_11), so pluralist test modules
import the model through ``import_pluralist`` (which restores the previous
setting) and switch float64 on only for their own tests via ``x64_only_here``.
"""

import jax
import pytest

PREVIOUS_X64 = bool(jax.config.jax_enable_x64)


def import_pluralist():
    """Import the package and leave float64 ON (the module-level switch only
    fires on first import); callers build their fixtures, then ``restore()``."""
    from src.pluralist import experiments, extended, model
    jax.config.update("jax_enable_x64", True)
    return model, experiments, extended


def restore():
    jax.config.update("jax_enable_x64", PREVIOUS_X64)


@pytest.fixture(autouse=True)
def x64_only_here():
    jax.config.update("jax_enable_x64", True)
    yield
    restore()

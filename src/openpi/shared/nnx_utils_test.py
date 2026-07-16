import concurrent.futures
import threading
import types

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np

from openpi.shared import nnx_utils


class _Affine(nnx.Module):
    def __init__(self):
        self.scale = nnx.Param(jnp.asarray(2.0, dtype=jnp.float32))

    def apply(self, x, *, bias=0.0, noise=None):
        del noise
        return x * self.scale + bias


def test_module_jit_compiled_executable_matches_standard_jit():
    model = _Affine()
    standard = nnx_utils.module_jit(model.apply)
    direct = nnx_utils.module_jit(model.apply, use_compiled_executable=True)

    for x in (
        jnp.arange(4, dtype=jnp.float32),
        jnp.asarray([5.0, -2.0, 3.5, 1.0], dtype=jnp.float32),
    ):
        np.testing.assert_array_equal(direct(x, bias=1.25), standard(x, bias=1.25))

    noise = jnp.ones((4,), dtype=jnp.float32)
    np.testing.assert_array_equal(
        direct(jnp.arange(4, dtype=jnp.float32), noise=noise),
        standard(jnp.arange(4, dtype=jnp.float32), noise=noise),
    )


def test_module_jit_compiled_executable_caches_by_full_signature(monkeypatch):
    compile_count = 0
    compiled_call_count = 0

    class _FakeLowered:
        def __init__(self, fun):
            self._fun = fun

        def compile(self):
            nonlocal compile_count
            compile_count += 1

            def compiled(*args, **kwargs):
                nonlocal compiled_call_count
                compiled_call_count += 1
                return self._fun(*args, **kwargs)

            return compiled

    class _FakeJitted:
        def __init__(self, fun):
            self._fun = fun

        def lower(self, *args, **kwargs):
            del args, kwargs
            return _FakeLowered(self._fun)

    monkeypatch.setattr(nnx_utils.jax, "jit", lambda fun, *args, **kwargs: _FakeJitted(fun))
    direct = nnx_utils.module_jit(_Affine().apply, use_compiled_executable=True)

    direct(jnp.ones((2,), dtype=jnp.float32), bias=1.0)
    direct(jnp.zeros((2,), dtype=jnp.float32), bias=1.0)
    assert compile_count == 1
    assert compiled_call_count == 2

    direct(jnp.ones((3,), dtype=jnp.float32), bias=1.0)
    direct(jnp.ones((3,), dtype=jnp.int32), bias=1.0)
    direct(jnp.ones((3,), dtype=jnp.int32), bias=2.0)
    direct(
        jnp.ones((3,), dtype=jnp.int32),
        bias=2.0,
        noise=jnp.ones((3,), dtype=jnp.float32),
    )
    direct(
        jnp.zeros((3,), dtype=jnp.int32),
        noise=jnp.zeros((3,), dtype=jnp.float32),
        bias=3.0,
    )
    assert compile_count == 4


def test_module_jit_compiles_concurrent_signature_once(monkeypatch):
    compile_count = 0
    compile_started = threading.Event()
    release_compile = threading.Event()

    class _FakeLowered:
        def __init__(self, fun):
            self._fun = fun

        def compile(self):
            nonlocal compile_count
            compile_count += 1
            compile_started.set()
            assert release_compile.wait(timeout=5)
            return self._fun

    class _FakeJitted:
        def __init__(self, fun):
            self._fun = fun

        def lower(self, *args, **kwargs):
            del args, kwargs
            return _FakeLowered(self._fun)

    monkeypatch.setattr(nnx_utils.jax, "jit", lambda fun, *args, **kwargs: _FakeJitted(fun))
    direct = nnx_utils.module_jit(_Affine().apply, use_compiled_executable=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(direct, jnp.ones((2,), dtype=jnp.float32))
        assert compile_started.wait(timeout=5)
        second = executor.submit(direct, jnp.zeros((2,), dtype=jnp.float32))
        release_compile.set()
        np.testing.assert_array_equal(first.result(timeout=5), jnp.full((2,), 2.0))
        np.testing.assert_array_equal(second.result(timeout=5), jnp.zeros((2,), dtype=jnp.float32))

    assert compile_count == 1


def test_module_jit_compiled_executable_falls_back_for_static_args(monkeypatch):
    lower_count = 0
    call_count = 0

    class _FakeJitted:
        def __init__(self, fun):
            self._fun = fun

        def __call__(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return self._fun(*args, **kwargs)

        def lower(self, *args, **kwargs):
            nonlocal lower_count
            del args, kwargs
            lower_count += 1
            raise AssertionError("direct lowering should not be used")

    monkeypatch.setattr(nnx_utils.jax, "jit", lambda fun, *args, **kwargs: _FakeJitted(fun))
    wrapped = nnx_utils.module_jit(
        _Affine().apply,
        use_compiled_executable=True,
        static_argnames=("bias",),
    )

    np.testing.assert_array_equal(wrapped(jnp.ones((2,), dtype=jnp.float32), bias=1.0), jnp.full((2,), 3.0))
    assert lower_count == 0
    assert call_count == 1


def test_module_jit_compiled_executable_falls_back_for_dynamic_shapes(monkeypatch):
    lower_count = 0
    call_count = 0

    class _FakeJitted:
        def __init__(self, fun):
            self._fun = fun

        def __call__(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return self._fun(*args, **kwargs)

        def lower(self, *args, **kwargs):
            nonlocal lower_count
            del args, kwargs
            lower_count += 1
            raise AssertionError("direct lowering should not be used")

    monkeypatch.setattr(nnx_utils.jax, "config", types.SimpleNamespace(jax_dynamic_shapes=True))
    monkeypatch.setattr(nnx_utils.jax, "jit", lambda fun, *args, **kwargs: _FakeJitted(fun))
    wrapped = nnx_utils.module_jit(_Affine().apply, use_compiled_executable=True)

    np.testing.assert_array_equal(wrapped(jnp.ones((2,), dtype=jnp.float32)), jnp.full((2,), 2.0))
    assert lower_count == 0
    assert call_count == 1

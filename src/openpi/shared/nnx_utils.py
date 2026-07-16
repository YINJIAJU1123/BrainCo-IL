from collections.abc import Callable
import dataclasses
import functools
import inspect
import logging
import re
import threading
from typing import Any, ParamSpec, TypeVar

import flax.nnx as nnx
import jax

from openpi.shared import array_typing as at

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)

_COMPILED_EXECUTABLE_CACHE_SIZE = 4
_UNSUPPORTED_COMPILED_EXECUTABLE_OPTIONS = (
    "static_argnums",
    "static_argnames",
    "donate_argnums",
    "donate_argnames",
    "abstracted_axes",
)


def module_jit(
    meth: Callable[P, R],
    *jit_args,
    use_compiled_executable: bool = False,
    **jit_kwargs,
) -> Callable[P, R]:
    """A higher-order function to JIT-compile `nnx.Module` methods, freezing the module's state in the process.

    Why not `nnx.jit`? For some reason, naively applying `nnx.jit` to `nnx.Module` methods, bound or unbound, uses much
    more memory than necessary. I'm guessing it has something to do with the fact that it must keep track of module
    mutations. Also, `nnx.jit` has some inherent overhead compared to a standard `jax.jit`, since every call must
    traverse the NNX module graph. See https://github.com/google/flax/discussions/4224 for details.

    `module_jit` is an alternative that avoids these issues by freezing the module's state. The function returned by
    `module_jit` acts exactly like the original method, except that the state of the module is frozen to whatever it was
    when `module_jit` was called. Mutations to the module within `meth` are still allowed, but they will be discarded
    after the method call completes.

    When `use_compiled_executable` is true, calls bypass the general `jax.jit` dispatch path and invoke a compiled
    executable directly. Executables are cached by input pytree structure (including kwargs), leaf shape and dtype,
    weak type, sharding, and layout. This mode is intended for inference with a small number of stable signatures.

    JAX AOT executables do not accept the original static arguments when called, and donation is incompatible with the
    frozen state being reused. Requests combining this mode with static arguments, donation, abstracted axes, or
    positional JIT options therefore fall back explicitly to the standard jax.jit call path. Direct mode also assumes
    that the JAX trace configuration, default device, and mesh remain stable for the lifetime of the wrapper.
    """
    if not (inspect.ismethod(meth) and isinstance(meth.__self__, nnx.Module)):
        raise ValueError("module_jit must only be used on bound methods of nnx.Modules.")

    graphdef, state = nnx.split(meth.__self__)

    def fun(state: nnx.State, *args: P.args, **kwargs: P.kwargs) -> R:
        module = nnx.merge(graphdef, state)
        return meth.__func__(module, *args, **kwargs)

    jitted_fn = jax.jit(fun, *jit_args, **jit_kwargs)

    unsupported_options = _unsupported_compiled_executable_options(jit_args, jit_kwargs)
    if use_compiled_executable and unsupported_options:
        logger.warning(
            "Direct compiled executable disabled for %s because of unsupported JIT options: %s",
            meth.__qualname__,
            ", ".join(unsupported_options),
        )
        use_compiled_executable = False
    if use_compiled_executable and jax.config.jax_dynamic_shapes:
        logger.warning(
            "Direct compiled executable disabled for %s because JAX dynamic shapes are enabled",
            meth.__qualname__,
        )
        use_compiled_executable = False

    if not use_compiled_executable:

        @functools.wraps(meth)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return jitted_fn(state, *args, **kwargs)

        return wrapper

    compiled_cache: dict[tuple[Any, ...], Any] = {}
    compile_lock = threading.Lock()
    direct_disabled = False
    cache_saturated = False

    @functools.wraps(meth)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        nonlocal cache_saturated, direct_disabled

        if direct_disabled:
            return jitted_fn(state, *args, **kwargs)
        signature = _input_signature(args, kwargs)
        if signature is None:
            return jitted_fn(state, *args, **kwargs)

        compiled_fn = compiled_cache.get(signature)
        if compiled_fn is not None:
            try:
                return compiled_fn(state, *args, **kwargs)
            except (TypeError, ValueError) as exc:
                with compile_lock:
                    compiled_cache.clear()
                    direct_disabled = True
                logger.warning(
                    "Direct compiled executable call for %s was incompatible; disabling direct mode: %s",
                    meth.__qualname__,
                    exc,
                )
                return jitted_fn(state, *args, **kwargs)

        if cache_saturated:
            return jitted_fn(state, *args, **kwargs)

        # Serialize compilation and the executable's first (lazily initialized) call. The executable is published only
        # after that first call succeeds, so concurrent callers cannot duplicate initialization or observe a bad entry.
        should_fallback = False
        with compile_lock:
            if direct_disabled or cache_saturated:
                should_fallback = True
            else:
                compiled_fn = compiled_cache.get(signature)
                if compiled_fn is not None:
                    try:
                        return compiled_fn(state, *args, **kwargs)
                    except (TypeError, ValueError) as exc:
                        compiled_cache.clear()
                        direct_disabled = True
                        should_fallback = True
                        logger.warning(
                            "Direct compiled executable call for %s was incompatible; disabling direct mode: %s",
                            meth.__qualname__,
                            exc,
                        )
                elif len(compiled_cache) >= _COMPILED_EXECUTABLE_CACHE_SIZE:
                    cache_saturated = True
                    should_fallback = True
                    logger.warning(
                        "Direct executable cache for %s reached %d signatures; future misses will use jax.jit",
                        meth.__qualname__,
                        _COMPILED_EXECUTABLE_CACHE_SIZE,
                    )
                elif not should_fallback:
                    logger.info(
                        "Compiling direct executable for %s (signature %d)",
                        meth.__qualname__,
                        len(compiled_cache) + 1,
                    )
                    # Lowering reconstructs pytree dataclasses with JAX ``ArgInfo`` placeholders. Runtime annotation
                    # checkers correctly reject those placeholders as non-arrays even though they are valid lowering
                    # inputs, so disable only those checks while JAX builds the lowered program.
                    with at.disable_typechecking():
                        compiled_fn = jitted_fn.lower(state, *args, **kwargs).compile()
                    try:
                        result = compiled_fn(state, *args, **kwargs)
                    except (TypeError, ValueError) as exc:
                        compiled_cache.clear()
                        direct_disabled = True
                        should_fallback = True
                        logger.warning(
                            "Direct compiled executable initialization for %s was incompatible; "
                            "disabling direct mode: %s",
                            meth.__qualname__,
                            exc,
                        )
                    else:
                        compiled_cache[signature] = compiled_fn
                        return result

        if should_fallback:
            return jitted_fn(state, *args, **kwargs)
        raise AssertionError("unreachable")

    return wrapper


def _unsupported_compiled_executable_options(jit_args: tuple[Any, ...], jit_kwargs: dict[str, Any]) -> list[str]:
    unsupported = ["positional JIT options"] if jit_args else []
    for name in _UNSUPPORTED_COMPILED_EXECUTABLE_OPTIONS:
        value = jit_kwargs.get(name)
        if value is None or (isinstance(value, (tuple, list, dict)) and not value):
            continue
        unsupported.append(name)
    return unsupported


def _input_signature(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...] | None:
    """Return a conservative, hashable signature for a lowered executable's dynamic call inputs."""
    try:
        leaves, treedef = jax.tree_util.tree_flatten((args, kwargs))
        signature = (treedef, tuple(_leaf_signature(leaf) for leaf in leaves))
        hash(signature)
    except (TypeError, ValueError):
        return None
    return signature


def _leaf_signature(leaf: Any) -> tuple[Any, ...]:
    aval = jax.core.get_aval(leaf)
    return (
        type(leaf),
        type(aval),
        aval,
        getattr(leaf, "sharding", None),
        getattr(leaf, "layout", None),
        getattr(leaf, "committed", None),
    )


@dataclasses.dataclass(frozen=True)
class PathRegex:
    """NNX Filter that matches paths using a regex.

    By default, paths are joined with a `/` separator. This can be overridden by setting the `sep` argument.
    """

    pattern: str | re.Pattern
    sep: str = "/"

    def __post_init__(self):
        if not isinstance(self.pattern, re.Pattern):
            object.__setattr__(self, "pattern", re.compile(self.pattern))

    def __call__(self, path: nnx.filterlib.PathParts, x: Any) -> bool:
        joined_path = self.sep.join(str(x) for x in path)
        assert isinstance(self.pattern, re.Pattern)
        return self.pattern.fullmatch(joined_path) is not None


def state_map(state: nnx.State, filter: nnx.filterlib.Filter, fn: Callable[[Any], Any]) -> nnx.State:
    """Apply a function to the leaves of the state that match the filter."""
    filtered_keys = set(state.filter(filter).flat_state())
    return state.map(lambda k, v: fn(v) if k in filtered_keys else v)

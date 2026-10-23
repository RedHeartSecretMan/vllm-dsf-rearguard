"""Execute selected functions, methods or classes straight from a source file.

The CPU checks run the production code of the adaptation layer without
importing the modules around it (CUDA kernels, vLLM engine, transformers).
Each helper parses the file, keeps only the requested definitions and executes
them in a namespace the caller supplies for the globals they use.

``annotations`` controls how type annotations are evaluated. With True the
definitions compile as if the file began with ``from __future__ import
annotations``, so annotations that name unavailable types stay unevaluated.
"""

from __future__ import annotations
import __future__

import ast
from pathlib import Path
from typing import Any

_FUTURE = __future__.annotations.compiler_flag


def _tree(path: str | Path) -> ast.Module:
    return ast.parse(Path(path).read_text())


def _execute(
    nodes: list[ast.stmt],
    path: str | Path,
    namespace: dict[str, Any],
    annotations: bool,
    filename: str | None = None,
) -> dict[str, Any]:
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    code = compile(
        module,
        filename or str(path),
        "exec",
        flags=_FUTURE if annotations else 0,
    )
    exec(code, namespace)
    return namespace


def load_functions(
    path: str | Path,
    names: tuple[str, ...] | list[str],
    namespace: dict[str, Any] | None = None,
    *,
    annotations: bool = True,
    anywhere: bool = False,
    filename: str | None = None,
) -> dict[str, Any]:
    """Run the named functions and return the namespace that now holds them.

    By default only top-level functions are considered. ``anywhere`` also
    accepts functions nested in classes or other functions.
    """
    tree = _tree(path)
    candidates = ast.walk(tree) if anywhere else tree.body
    selected = [
        node
        for node in candidates
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in selected} != set(names):
        raise AssertionError(f"Missing source function in {path}: {sorted(names)}")
    return _execute(selected, path, dict(namespace or {}), annotations, filename)


def load_function(
    path: str | Path, name: str, namespace: dict[str, Any] | None = None, **options: Any
) -> Any:
    return load_functions(path, (name,), namespace, **options)[name]


def find_method(path: str | Path, cls: str, name: str) -> ast.FunctionDef:
    owner = next(
        node
        for node in _tree(path).body
        if isinstance(node, ast.ClassDef) and node.name == cls
    )
    return next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def load_method(
    path: str | Path,
    cls: str,
    name: str,
    namespace: dict[str, Any],
    *,
    annotations: bool = True,
) -> Any:
    """Run one method of ``cls`` as a plain function taking ``self`` first."""
    method = find_method(path, cls, name)
    return _execute([method], path, namespace, annotations)[name]


def run_function_tree(
    module: ast.Module,
    path: str | Path,
    namespace: dict[str, Any],
    name: str,
    *,
    annotations: bool = True,
) -> Any:
    """Run a module the caller built from source nodes and return ``name``."""
    return _execute(module.body, path, namespace, annotations)[name]


def load_class(
    path: str | Path,
    namespace: dict[str, Any],
    *,
    bases: list[str] | None = None,
    annotations: bool = False,
    filename: str | None = None,
) -> Any:
    """Run the first class of the file, optionally replacing its base classes."""
    node = next(n for n in _tree(path).body if isinstance(n, ast.ClassDef))
    if bases is not None:
        node.bases = [ast.Name(id=base, ctx=ast.Load()) for base in bases]
    _execute([node], path, namespace, annotations, filename)
    return namespace[node.name]

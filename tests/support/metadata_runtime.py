# CPU test support; original notices in NOTICE.
import ast
from pathlib import Path

def metadata_prefix(path, cls, kind):
    """Execute the actual changed builder prefix, stopping before unrelated kernels.

    Indexer stops at the first prefill assertion. SWA stops after validity is
    written. This tests the production branch and persistent output addresses;
    it does not claim to execute an entire attention builder on a CPU.
    """
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    method = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == 'build')
    prefix = []
    for statement in method.body:
        if kind == 'indexer' and isinstance(statement, ast.If) and ast.unparse(statement.test) == 'num_prefills > 0':
            statement.body = statement.body[:next(i for i, s in enumerate(statement.body) if isinstance(s, ast.Assert))]
            # The original variant has no statements before its assertion.
            if not statement.body:
                statement.body = [ast.Pass()]
            prefix.append(statement)
            break
        if kind == 'swa' and isinstance(statement, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'non_causal' for t in statement.targets):
            break
        prefix.append(statement)
    else:
        raise ValueError('Builder prefix boundary changed')
    method.body = prefix + [ast.Return(value=ast.Call(func=ast.Name(id='locals', ctx=ast.Load()), args=[], keywords=[]))]
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    return module

"""Exercise target allocation and the V2 runner's optional-buffer contract."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from scripts.source_extract import find_method, load_method, run_function_tree

SOURCE = Path(__file__).resolve().parents[1] / "adaptation/files/vllm"
MODEL = SOURCE / "models/deepseek_v4_1/nvidia/model.py"
RUNNER = SOURCE / "v1/worker/gpu/model_runner.py"


class DSparkMTPBufferTests(unittest.TestCase):
    def allocate(self, method, *, last_pp=True):
        # Use the production constructor's final allocation block without
        # importing the GPU model or allocating its weights.
        init = find_method(MODEL, "DeepseekV4Model", "__init__")
        start = next(
            i
            for i, statement in enumerate(init.body)
            if isinstance(statement, ast.Assign)
            and any(
                isinstance(name, ast.Name) and name.id == "spec_config"
                for name in statement.targets
            )
        )
        spec = (
            None
            if method is None
            else SimpleNamespace(
                use_eagle=lambda: method in {"mtp", "eagle3", "dspark"},
                uses_draft_model=lambda: method == "draft_model",
                use_dspark=lambda: method == "dspark",
            )
        )
        owner = SimpleNamespace(hc_dim=20480)
        torch = SimpleNamespace(empty=Mock(return_value=object()))
        namespace = {
            "self": owner,
            "torch": torch,
            "get_pp_group": lambda: SimpleNamespace(is_last_rank=last_pp),
            "vllm_config": SimpleNamespace(
                speculative_config=spec,
                scheduler_config=SimpleNamespace(max_num_batched_tokens=16384),
                model_config=SimpleNamespace(dtype="bf16"),
            ),
        }
        run_function_tree(
            ast.Module(body=init.body[start:], type_ignores=[]),
            MODEL,
            namespace,
            "self",
        )
        return owner._mtp_hidden_buffer, torch.empty

    def test_dspark_does_not_reserve_the_mtp_residual(self):
        for method, last_pp in (("dspark", True), (None, True), ("mtp", False)):
            with self.subTest(method=method, last_pp=last_pp):
                buffer, allocation = self.allocate(method, last_pp=last_pp)
                self.assertIsNone(buffer)
                allocation.assert_not_called()

    def test_other_draft_modes_keep_existing_reservation(self):
        for method in ("mtp", "eagle3", "draft_model"):
            with self.subTest(method=method):
                buffer, allocation = self.allocate(method)
                self.assertIs(buffer, allocation.return_value)
                allocation.assert_called_once_with(16384, 20480, dtype="bf16")

    def test_v2_uses_active_mtp_rows_or_the_forward_result(self):
        choose = load_method(RUNNER, "GPUModelRunner", "_get_draft_hidden_states", {})
        hidden_states = SimpleNamespace(shape=(3, 5120))
        for model in (
            SimpleNamespace(),
            SimpleNamespace(get_mtp_target_hidden_states=lambda: None),
        ):
            with self.subTest(model=model):
                self.assertIs(
                    choose(SimpleNamespace(model=model), hidden_states), hidden_states
                )
        pre_collapse = [object() for _ in range(5)]
        model = SimpleNamespace(get_mtp_target_hidden_states=lambda: pre_collapse)
        self.assertEqual(
            choose(SimpleNamespace(model=model), hidden_states), pre_collapse[:3]
        )

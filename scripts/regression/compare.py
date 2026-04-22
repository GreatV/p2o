from __future__ import annotations

from typing import Any


def canonical_row_order(key, sort_columns: list[int] | None, round_decimals: int) -> Any:
    import numpy as np

    local_sort_columns = sort_columns
    if local_sort_columns is None:
        local_sort_columns = list(range(key.shape[1]))
    if not isinstance(local_sort_columns, list) or not all(isinstance(col, int) for col in local_sort_columns):
        raise ValueError("compare_config.sort_columns must be an int list")

    return np.lexsort(tuple(np.round(key[:, col], round_decimals) for col in reversed(local_sort_columns)))


def greedy_row_assignment(
    paddle_key,
    onnx_key,
    match_columns: list[int] | None,
    round_decimals: int,
    exact_columns: list[int] | None = None,
) -> list[int]:
    import numpy as np

    if paddle_key.ndim != 2 or onnx_key.ndim != 2:
        raise ValueError("instance_row_match requires 2D key outputs")
    if paddle_key.shape[0] != onnx_key.shape[0]:
        raise ValueError("instance_row_match requires Paddle and ONNX row counts to match")

    local_match_columns = match_columns
    if local_match_columns is None:
        local_match_columns = list(range(paddle_key.shape[1]))
    if not isinstance(local_match_columns, list) or not all(isinstance(col, int) for col in local_match_columns):
        raise ValueError("compare_config.match_columns must be an int list")

    local_exact_columns = exact_columns or []
    if not isinstance(local_exact_columns, list) or not all(isinstance(col, int) for col in local_exact_columns):
        raise ValueError("compare_config.exact_columns must be an int list")

    paddle_view = np.round(paddle_key[:, local_match_columns].astype(np.float64, copy=False), round_decimals)
    onnx_view = np.round(onnx_key[:, local_match_columns].astype(np.float64, copy=False), round_decimals)

    scales = np.ptp(np.concatenate([paddle_view, onnx_view], axis=0), axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    costs = np.abs(paddle_view[:, None, :] - onnx_view[None, :, :]) / scales[None, None, :]
    total_cost = np.sum(costs, axis=2)

    for col in local_exact_columns:
        if col not in local_match_columns:
            continue
        col_idx = local_match_columns.index(col)
        total_cost += np.where(paddle_view[:, None, col_idx] == onnx_view[None, :, col_idx], 0.0, 1e6)

    chosen = np.full(paddle_key.shape[0], -1, dtype=np.int64)
    used_onnx = np.zeros(onnx_key.shape[0], dtype=bool)
    pair_order = np.argsort(total_cost, axis=None)
    for flat_idx in pair_order:
        paddle_idx, onnx_idx = np.unravel_index(flat_idx, total_cost.shape)
        if chosen[paddle_idx] != -1 or used_onnx[onnx_idx]:
            continue
        chosen[paddle_idx] = onnx_idx
        used_onnx[onnx_idx] = True
        if np.all(chosen != -1):
            break

    if np.any(chosen == -1):
        raise RuntimeError("Failed to build a complete row assignment for instance_row_match")
    return chosen.tolist()


def group_equal_rows(key, group_columns: list[int] | None, round_decimals: int) -> list[tuple[int, int]]:
    import numpy as np

    if key.ndim != 2:
        raise ValueError("tie-aware grouping requires a 2D key output")

    local_group_columns = group_columns
    if local_group_columns is None:
        local_group_columns = list(range(key.shape[1]))
    if not isinstance(local_group_columns, list) or not all(isinstance(col, int) for col in local_group_columns):
        raise ValueError("compare_config.tie_group_columns must be an int list")

    rounded = np.round(key[:, local_group_columns].astype(np.float64, copy=False), round_decimals)
    groups: list[tuple[int, int]] = []
    start = 0
    while start < rounded.shape[0]:
        end = start + 1
        while end < rounded.shape[0] and np.array_equal(rounded[start], rounded[end]):
            end += 1
        groups.append((start, end))
        start = end
    return groups


def output_row_cost_matrix(paddle_group, onnx_group) -> Any:
    import numpy as np

    if paddle_group.shape[0] != onnx_group.shape[0]:
        raise ValueError("tie-aware output matching requires equal group sizes")

    if np.issubdtype(paddle_group.dtype, np.integer) or np.issubdtype(paddle_group.dtype, np.bool_):
        paddle_view = paddle_group.astype(np.int64, copy=False)
        onnx_view = onnx_group.astype(np.int64, copy=False)
    else:
        paddle_view = paddle_group.astype(np.float64, copy=False)
        onnx_view = onnx_group.astype(np.float64, copy=False)

    paddle_flat = paddle_view.reshape(paddle_view.shape[0], -1)
    onnx_flat = onnx_view.reshape(onnx_view.shape[0], -1)
    return np.mean(np.abs(paddle_flat[:, None, :] - onnx_flat[None, :, :]), axis=2)


def greedy_group_assignment(cost_matrix) -> list[int]:
    import numpy as np

    if cost_matrix.shape[0] != cost_matrix.shape[1]:
        raise ValueError("tie-aware output matching requires a square cost matrix")

    chosen = np.full(cost_matrix.shape[0], -1, dtype=np.int64)
    used = np.zeros(cost_matrix.shape[1], dtype=bool)
    for flat_idx in np.argsort(cost_matrix, axis=None):
        paddle_idx, onnx_idx = np.unravel_index(flat_idx, cost_matrix.shape)
        if chosen[paddle_idx] != -1 or used[onnx_idx]:
            continue
        chosen[paddle_idx] = onnx_idx
        used[onnx_idx] = True
        if np.all(chosen != -1):
            break

    if np.any(chosen == -1):
        raise RuntimeError("Failed to build a complete tie-aware group assignment")
    return chosen.tolist()


def refine_tie_group_outputs(
    normalized_paddle: list[Any],
    normalized_onnx: list[Any],
    key_output: int,
    tie_group_columns: list[int] | None,
    tie_aware_outputs: list[int],
    round_decimals: int,
) -> tuple[list[Any], list[Any]]:
    if not tie_aware_outputs:
        return normalized_paddle, normalized_onnx

    key = normalized_paddle[key_output]
    groups = group_equal_rows(key, tie_group_columns, round_decimals)
    if not groups:
        return normalized_paddle, normalized_onnx

    refined_paddle = list(normalized_paddle)
    refined_onnx = list(normalized_onnx)
    for start, end in groups:
        if end - start <= 1:
            continue

        total_cost = None
        for output_idx in tie_aware_outputs:
            cost = output_row_cost_matrix(refined_paddle[output_idx][start:end], refined_onnx[output_idx][start:end])
            total_cost = cost if total_cost is None else total_cost + cost

        if total_cost is None:
            continue

        local_order = greedy_group_assignment(total_cost)
        for output_idx in tie_aware_outputs:
            refined_onnx[output_idx][start:end] = refined_onnx[output_idx][start:end][local_order]
    return refined_paddle, refined_onnx


def normalize_outputs(
    paddle_outputs: list[Any],
    onnx_outputs: list[Any],
    compare_config: dict[str, Any] | None,
) -> tuple[list[Any], list[Any]]:
    if not compare_config:
        return paddle_outputs, onnx_outputs

    mode = compare_config.get("mode")
    if mode not in {"instance_row_sort", "instance_row_match", "pairwise_order_match", "permutation_match"}:
        return paddle_outputs, onnx_outputs

    key_output = int(compare_config.get("key_output", 0))
    reorder_outputs = compare_config.get("reorder_outputs", [key_output])
    if not isinstance(reorder_outputs, list) or not all(isinstance(idx, int) for idx in reorder_outputs):
        raise ValueError("compare_config.reorder_outputs must be an int list")

    round_decimals = int(compare_config.get("round_decimals", 6))
    sort_columns = compare_config.get("sort_columns")
    if mode in {"instance_row_match", "pairwise_order_match", "permutation_match"}:
        match_columns = compare_config.get("match_columns", sort_columns)
        exact_columns = compare_config.get("exact_columns", [])

        paddle_key = paddle_outputs[key_output]
        onnx_key = onnx_outputs[key_output]
        if paddle_key.ndim != 2 or onnx_key.ndim != 2:
            raise ValueError(f"{mode} requires a 2D key output")

        paddle_order = canonical_row_order(paddle_key, sort_columns, round_decimals)
        sorted_paddle_key = paddle_key[paddle_order]
        onnx_order = greedy_row_assignment(
            sorted_paddle_key,
            onnx_key,
            match_columns=match_columns,
            round_decimals=round_decimals,
            exact_columns=exact_columns,
        )

        normalized_paddle = list(paddle_outputs)
        normalized_onnx = list(onnx_outputs)
        for output_idx in reorder_outputs:
            normalized_paddle[output_idx] = normalized_paddle[output_idx][paddle_order]
            normalized_onnx[output_idx] = normalized_onnx[output_idx][onnx_order]
        if mode in {"pairwise_order_match", "permutation_match"}:
            return normalized_paddle, normalized_onnx

        tie_group_columns = compare_config.get("tie_group_columns", match_columns)
        tie_aware_outputs = compare_config.get("tie_aware_outputs", [])
        if not isinstance(tie_aware_outputs, list) or not all(isinstance(idx, int) for idx in tie_aware_outputs):
            raise ValueError("compare_config.tie_aware_outputs must be an int list")
        return refine_tie_group_outputs(
            normalized_paddle,
            normalized_onnx,
            key_output=key_output,
            tie_group_columns=tie_group_columns,
            tie_aware_outputs=tie_aware_outputs,
            round_decimals=round_decimals,
        )

    def reorder(outputs: list[Any]) -> list[Any]:
        key = outputs[key_output]
        if key.ndim != 2:
            raise ValueError("instance_row_sort requires a 2D key output")

        order = canonical_row_order(key, sort_columns, round_decimals)
        normalized = list(outputs)
        for output_idx in reorder_outputs:
            normalized[output_idx] = normalized[output_idx][order]
        return normalized

    return reorder(paddle_outputs), reorder(onnx_outputs)


def compare_pairwise_order_columns(
    paddle_output,
    onnx_output,
    columns: list[int],
    round_decimals: int,
) -> dict[str, Any]:
    import numpy as np

    if paddle_output.ndim != 2 or onnx_output.ndim != 2:
        raise ValueError("pairwise_order_columns only supports 2D outputs")
    if paddle_output.shape != onnx_output.shape:
        raise ValueError("pairwise_order_columns requires matching output shapes")
    if not isinstance(columns, list) or not columns or not all(isinstance(col, int) for col in columns):
        raise ValueError("pairwise_order_columns must be a non-empty int list")

    mask = ~np.eye(paddle_output.shape[0], dtype=bool)
    column_reports = []
    passed = True
    for col in columns:
        if col < 0 or col >= paddle_output.shape[1]:
            raise ValueError(f"pairwise_order_columns contains out-of-range column {col}")

        paddle_col = np.round(paddle_output[:, col].astype(np.float64, copy=False), round_decimals)
        onnx_col = np.round(onnx_output[:, col].astype(np.float64, copy=False), round_decimals)
        paddle_rel = np.sign(paddle_col[:, None] - paddle_col[None, :]).astype(np.int8, copy=False)
        onnx_rel = np.sign(onnx_col[:, None] - onnx_col[None, :]).astype(np.int8, copy=False)
        agreement = paddle_rel == onnx_rel
        agreement_ratio = float(np.mean(agreement[mask])) if agreement.size else 1.0
        tie_mask = paddle_rel == 0
        tie_count = int(np.sum(tie_mask[mask]))
        mismatch_count = int(np.sum(~agreement[mask]))
        col_passed = bool(np.all(agreement[mask]))
        column_reports.append(
            {
                "column": col,
                "pairwise_agreement": agreement_ratio,
                "mismatch_count": mismatch_count,
                "tie_count": tie_count,
                "passed": col_passed,
            }
        )
        passed = passed and col_passed

    return {"columns": column_reports, "passed": passed}


def compare_permutation_columns(
    paddle_output,
    onnx_output,
    columns: list[int],
    round_decimals: int,
) -> dict[str, Any]:
    import numpy as np

    if paddle_output.ndim != 2 or onnx_output.ndim != 2:
        raise ValueError("permutation_columns only supports 2D outputs")
    if paddle_output.shape != onnx_output.shape:
        raise ValueError("permutation_columns requires matching output shapes")
    if not isinstance(columns, list) or not columns or not all(isinstance(col, int) for col in columns):
        raise ValueError("permutation_columns must be a non-empty int list")

    column_reports = []
    passed = True
    for col in columns:
        if col < 0 or col >= paddle_output.shape[1]:
            raise ValueError(f"permutation_columns contains out-of-range column {col}")

        paddle_col = np.round(paddle_output[:, col].astype(np.float64, copy=False), round_decimals)
        onnx_col = np.round(onnx_output[:, col].astype(np.float64, copy=False), round_decimals)
        paddle_perm = np.argsort(paddle_col, kind="stable")
        onnx_perm = np.argsort(onnx_col, kind="stable")
        mismatch_positions = np.flatnonzero(paddle_perm != onnx_perm)
        first_examples = []
        for pos in mismatch_positions[:10]:
            paddle_row = int(paddle_perm[pos])
            onnx_row = int(onnx_perm[pos])
            first_examples.append(
                {
                    "position": int(pos),
                    "paddle_row": paddle_row,
                    "onnx_row": onnx_row,
                    "paddle_value": float(paddle_col[paddle_row]),
                    "onnx_value": float(onnx_col[onnx_row]),
                }
            )

        col_passed = mismatch_positions.size == 0
        column_reports.append(
            {
                "column": col,
                "mismatch_count": int(mismatch_positions.size),
                "first_mismatch_positions": mismatch_positions[:10].astype(int).tolist(),
                "first_mismatch_examples": first_examples,
                "passed": col_passed,
            }
        )
        passed = passed and col_passed

    return {"columns": column_reports, "passed": passed}


def compare_outputs(
    paddle_outputs: list[Any],
    onnx_outputs: list[Any],
    rtol: float,
    atol: float,
    compare_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import numpy as np

    paddle_outputs, onnx_outputs = normalize_outputs(paddle_outputs, onnx_outputs, compare_config)
    per_output = compare_config.get("per_output", {}) if compare_config else {}
    round_decimals = int(compare_config.get("round_decimals", 6)) if compare_config else 6

    def output_rule(index: int) -> dict[str, Any]:
        rule = per_output.get(str(index), per_output.get(index, {}))
        if rule is None:
            return {}
        if not isinstance(rule, dict):
            raise ValueError(f"compare_config.per_output[{index}] must be an object")
        return rule

    report: dict[str, Any] = {
        "output_count_match": len(paddle_outputs) == len(onnx_outputs),
        "outputs": [],
    }
    if len(paddle_outputs) != len(onnx_outputs):
        report["passed"] = False
        return report

    passed = True
    for index, (paddle_output, onnx_output) in enumerate(zip(paddle_outputs, onnx_outputs)):
        rule = output_rule(index)
        item: dict[str, Any] = {
            "index": index,
            "paddle_shape": list(paddle_output.shape),
            "onnx_shape": list(onnx_output.shape),
            "paddle_dtype": str(paddle_output.dtype),
            "onnx_dtype": str(onnx_output.dtype),
            "shape_match": tuple(paddle_output.shape) == tuple(onnx_output.shape),
        }
        if not item["shape_match"]:
            item["passed"] = False
            passed = False
            report["outputs"].append(item)
            continue

        compare_paddle = paddle_output
        compare_onnx = onnx_output
        ignored_columns = rule.get("ignore_columns", [])
        pairwise_order_columns = rule.get("pairwise_order_columns", [])
        if pairwise_order_columns:
            if not isinstance(pairwise_order_columns, list) or not all(isinstance(col, int) for col in pairwise_order_columns):
                raise ValueError(f"compare_config.per_output[{index}].pairwise_order_columns must be an int list")
            ignored_columns = sorted(set(ignored_columns) | set(pairwise_order_columns))
        permutation_columns = rule.get("permutation_columns", [])
        if permutation_columns:
            if not isinstance(permutation_columns, list) or not all(isinstance(col, int) for col in permutation_columns):
                raise ValueError(f"compare_config.per_output[{index}].permutation_columns must be an int list")
            ignored_columns = sorted(set(ignored_columns) | set(permutation_columns))
        if ignored_columns:
            if compare_paddle.ndim != 2 or compare_onnx.ndim != 2:
                raise ValueError(f"ignore_columns only supports 2D outputs, got output {index}")
            if not isinstance(ignored_columns, list) or not all(isinstance(col, int) for col in ignored_columns):
                raise ValueError(f"compare_config.per_output[{index}].ignore_columns must be an int list")
            kept_columns = [col for col in range(compare_paddle.shape[1]) if col not in ignored_columns]
            compare_paddle = compare_paddle[:, kept_columns]
            compare_onnx = compare_onnx[:, kept_columns]

        if np.issubdtype(paddle_output.dtype, np.integer) or np.issubdtype(paddle_output.dtype, np.bool_):
            diff = compare_paddle.astype(np.int64, copy=False) - compare_onnx.astype(np.int64, copy=False)
            max_abs = int(np.max(np.abs(diff))) if diff.size else 0
            mean_abs = float(np.mean(np.abs(diff))) if diff.size else 0.0
            int_tolerance = rule.get("int_tolerance", {})
            max_abs_limit = int_tolerance.get("max_abs")
            mean_abs_limit = int_tolerance.get("mean_abs")
            passed_output = max_abs == 0
            if max_abs_limit is not None or mean_abs_limit is not None:
                if max_abs_limit is not None and not isinstance(max_abs_limit, (int, float)):
                    raise ValueError(f"compare_config.per_output[{index}].int_tolerance.max_abs must be numeric")
                if mean_abs_limit is not None and not isinstance(mean_abs_limit, (int, float)):
                    raise ValueError(f"compare_config.per_output[{index}].int_tolerance.mean_abs must be numeric")
                passed_output = True
                if max_abs_limit is not None:
                    passed_output = passed_output and max_abs <= int(max_abs_limit)
                if mean_abs_limit is not None:
                    passed_output = passed_output and mean_abs <= float(mean_abs_limit)
            item.update(
                {
                    "max_abs_diff": max_abs,
                    "mean_abs_diff": mean_abs,
                    "passed": passed_output,
                }
            )
        else:
            output_rtol = float(rule.get("rtol", rtol))
            output_atol = float(rule.get("atol", atol))
            abs_diff = np.abs(compare_paddle.astype(np.float64) - compare_onnx.astype(np.float64))
            max_abs = float(np.max(abs_diff)) if abs_diff.size else 0.0
            mean_abs = float(np.mean(abs_diff)) if abs_diff.size else 0.0
            item.update(
                {
                    "max_abs_diff": max_abs,
                    "mean_abs_diff": mean_abs,
                    "passed": bool(np.allclose(compare_paddle, compare_onnx, rtol=output_rtol, atol=output_atol)),
                }
            )

        if pairwise_order_columns:
            pairwise_report = compare_pairwise_order_columns(
                paddle_output,
                onnx_output,
                columns=pairwise_order_columns,
                round_decimals=round_decimals,
            )
            item["pairwise_order"] = pairwise_report["columns"]
            item["passed"] = item["passed"] and pairwise_report["passed"]
        if permutation_columns:
            permutation_report = compare_permutation_columns(
                paddle_output,
                onnx_output,
                columns=permutation_columns,
                round_decimals=round_decimals,
            )
            item["permutation_order"] = permutation_report["columns"]
            item["passed"] = item["passed"] and permutation_report["passed"]
        passed = passed and item["passed"]
        report["outputs"].append(item)

    report["passed"] = passed
    return report

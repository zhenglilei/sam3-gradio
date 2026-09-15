"""Edge-constrained candidates and joint four-tile translation matching."""

import itertools
import math

import cv2
import numpy as np


def solve_grid_candidates(edges, tolerance):
    if len(edges) != 4 or any(not edge for edge in edges):
        return None
    best = None
    best_key = None
    for choice in itertools.product(*edges):
        top, left, right, bottom = choice
        error = math.hypot(top[0] + right[0] - left[0] - bottom[0],
                           top[1] + right[1] - left[1] - bottom[1])
        if error > tolerance:
            continue
        key = (sum(item[2] for item in choice), -error)
        if best_key is None or key > best_key:
            best, best_key = choice, key
    return best


def match_grid_pair(a, b, axis):
    from .stitch_workflow import (
        _alignment_evidence,
        _alignment_quality,
        _alignment_score,
        _minimum_primary_extent,
        _resize_alignment_pair,
        detect_period,
    )

    height, width = a[0].shape
    ar, br, sx, sy = _resize_alignment_pair(a[0], b[0], 192)
    ah, bh, _, _ = _resize_alignment_pair(a[1], b[1], 192)
    small_a, small_b = (ar, ah), (br, bh)
    primary_size = height if axis == "vertical" else width
    cross_size = min(width, b[0].shape[1]) if axis == "vertical" else min(height, b[0].shape[0])
    primary_scale, cross_scale = (sy, sx) if axis == "vertical" else (sx, sy)
    px, py, _, _ = detect_period(a[0])
    axis_period = py if axis == "vertical" else px
    period_guard = axis_period if axis_period < 0.20 * primary_size else 0.0
    minimum_primary = _minimum_primary_extent(primary_size, period_guard)
    primary_lo = math.ceil(minimum_primary * primary_scale)
    primary_hi = math.floor((primary_size - 8) * primary_scale)
    cross_limit = math.floor(0.1 * cross_size * cross_scale)
    ranked = []
    for primary in range(primary_lo, primary_hi + 1):
        for cross in range(-cross_limit, cross_limit + 1):
            dx, dy = (cross, primary) if axis == "vertical" else (primary, cross)
            score = _alignment_score(small_a, small_b, dx, dy)
            ranked.append((score, int(round(dx / sx)), int(round(dy / sy))))
    ranked.sort(key=lambda item: (-item[0], item[2], item[1]))
    separation = max(8.0, 0.3 * axis_period)
    seeds = []
    for score, dx, dy in ranked:
        if score < 0.2:
            break
        if all(math.hypot(dx - x, dy - y) >= separation for x, y in seeds):
            seeds.append((dx, dy))
        if len(seeds) == 24:
            break
    result = []
    radius_x, radius_y = math.ceil(1 / sx), math.ceil(1 / sy)
    for x, y in seeds:
        best = None
        for dy in range(y - radius_y, y + radius_y + 1):
            for dx in range(x - radius_x, x + radius_x + 1):
                primary, cross = (dy, dx) if axis == "vertical" else (dx, dy)
                if not (minimum_primary <= primary <= primary_size - 8
                        and abs(cross) <= 0.1 * cross_size):
                    continue
                score = _alignment_score(a, b, dx, dy)
                if best is None or score > best[2]:
                    best = (dx, dy, score)
        if best is not None and best[2] >= 0.35:
            result.append(best)

    ranked_result = []
    seen = set()
    for dx, dy, score in result:
        key = (dx, dy)
        if key in seen:
            continue
        seen.add(key)
        evidence = _alignment_evidence(a, b, dx, dy)
        if not evidence["available"]:
            continue
        quality = _alignment_quality(score, evidence)
        ranked_result.append((quality, score, dx, dy))
    ranked_result.sort(key=lambda item: (-item[0], -item[1], item[3], item[2]))
    return [(dx, dy, quality) for quality, _score, dx, dy in ranked_result[:8]]


def _grid_tolerance(prepared):
    return max(4.0, 0.025 * min(min(item[0].shape[:2]) for item in prepared))


def _fetch_grid_candidates(cache, prepared, source, target, axis):
    key = (source, target, axis)
    if key not in cache:
        candidates = match_grid_pair(prepared[source], prepared[target], axis)
        cache[key] = list(candidates)[:8] if candidates is not None else []
    return cache[key]


def _edge_direction_error(prepared, source, target, dx, dy, axis):
    from .stitch_workflow import _minimum_primary_extent, detect_period

    source_shape = prepared[source][0].shape[:2]
    target_shape = prepared[target][0].shape[:2]
    if axis == "vertical":
        extent = source_shape[0]
        cross_extent = min(source_shape[1], target_shape[1])
        primary, cross = dy, dx
        period = detect_period(prepared[source][0])[1]
    else:
        extent = source_shape[1]
        cross_extent = min(source_shape[0], target_shape[0])
        primary, cross = dx, dy
        period = detect_period(prepared[source][0])[0]
    if float(np.std(prepared[source][0])) < 3.0:
        period = 0.0

    tolerance = 2.0
    if period >= 0.20 * extent:
        period = 0.0
    minimum_primary = _minimum_primary_extent(extent, period)
    if primary < minimum_primary - tolerance:
        return (
            f"主方向 {primary:.1f}px 小于邻边拼接最小步长 "
            f"{minimum_primary:.1f}px"
        )
    if primary > extent + tolerance:
        return f"主方向 {primary:.1f}px 超过前图尺寸 {extent}px"
    if abs(cross) > 0.1 * cross_extent + tolerance:
        return f"交叉方向 {cross:.1f}px 超过 10% 最小交叉尺寸 {0.1 * cross_extent:.1f}px"
    return None


def _joint_least_squares(prepared, edge_specs):
    """Solve all selected directed edge translations with tile 0 anchored."""
    tile_count = len(prepared)
    if tile_count == 1:
        return [(0, 0)], None
    if not edge_specs:
        return None, "没有足够的接缝约束，无法求解联合位置"

    incidence = np.zeros((len(edge_specs), tile_count - 1), dtype=float)
    observed = np.zeros((len(edge_specs), 2), dtype=float)
    for row, (label, source, target, axis, edge) in enumerate(edge_specs):
        if len(edge) < 3:
            return None, f"接缝 {label} 的候选格式无效"
        dx, dy, score = edge[:3]
        try:
            dx, dy, score = float(dx), float(dy), float(score)
        except (TypeError, ValueError):
            return None, f"接缝 {label} 的候选包含非数值"
        if not all(math.isfinite(value) for value in (dx, dy, score)):
            return None, f"接缝 {label} 的候选包含非有限值"
        if target != 0:
            incidence[row, target - 1] += 1.0
        if source != 0:
            incidence[row, source - 1] -= 1.0
        observed[row] = (dx, dy)

    if np.linalg.matrix_rank(incidence) < tile_count - 1:
        return None, "接缝图不连通，无法安全求解所有图片位置"
    solved, _, _, _ = np.linalg.lstsq(incidence, observed, rcond=None)
    if not np.all(np.isfinite(solved)):
        return None, "联合最小二乘产生了非有限位置"
    positions = [(0, 0)] + [
        (int(round(float(row[0]))), int(round(float(row[1]))))
        for row in solved
    ]
    for label, source, target, axis, _edge in edge_specs:
        dx = positions[target][0] - positions[source][0]
        dy = positions[target][1] - positions[source][1]
        error = _edge_direction_error(prepared, source, target, dx, dy, axis)
        if error is not None:
            return None, f"圆整后接缝 {label} 越界：{error}"
    return positions, None


def _topology_edge_specs(mapping):
    return [
        ("上边", mapping[0], mapping[1], "horizontal"),
        ("左边", mapping[0], mapping[2], "vertical"),
        ("右边", mapping[1], mapping[3], "vertical"),
        ("下边", mapping[2], mapping[3], "horizontal"),
    ]


def _solve_grid_topology(prepared, mapping, cache, tolerance):
    specs = _topology_edge_specs(mapping)
    edges = [
        _fetch_grid_candidates(cache, prepared, source, target, axis)
        for _label, source, target, axis in specs
    ]
    if any(not edge for edge in edges):
        return None

    best = None
    for choice in itertools.product(*edges):
        top, left, right, bottom = choice
        closure = math.hypot(
            top[0] + right[0] - left[0] - bottom[0],
            top[1] + right[1] - left[1] - bottom[1],
        )
        if closure > tolerance:
            continue
        edge_specs = [
            (spec[0], spec[1], spec[2], spec[3], edge)
            for spec, edge in zip(specs, choice)
        ]
        positions, validation_error = _joint_least_squares(
            prepared,
            edge_specs,
        )
        if positions is None:
            continue
        qualities = [float(edge[2]) for edge in choice]
        joint_quality = sum(qualities)
        key = (joint_quality, sum(float(edge[2]) for edge in choice), -closure)
        if best is None or key > best["key"]:
            best = {
                "mapping": mapping,
                "choice": choice,
                "edge_specs": edge_specs,
                "positions": positions,
                "closure": closure,
                "quality": joint_quality,
                "key": key,
            }
    return best


def _has_texture(prepared):
    return sum(float(np.std(item[0])) > 3.0 for item in prepared) >= 2


def align_four_tiles(prepared):
    tolerance = _grid_tolerance(prepared)
    cache = {}
    row_mapping = (0, 1, 2, 3)
    snake_mapping = (0, 1, 3, 2)
    solutions = []
    row_solution = _solve_grid_topology(
        prepared, row_mapping, cache, tolerance
    )
    if row_solution is not None:
        solutions.append(row_solution)
    if _has_texture(prepared):
        snake_solution = _solve_grid_topology(
            prepared, snake_mapping, cache, tolerance
        )
        if snake_solution is not None:
            solutions.append(snake_solution)
    if not solutions:
        return None, [
            "四条接缝在主方向重叠不超过 35% 的范围内没有一致解，保留规则网格；未完成配准。"
        ]

    if len(solutions) == 1:
        chosen = solutions[0]
    else:
        solutions.sort(key=lambda item: item["key"], reverse=True)
        chosen, alternate = solutions[0], solutions[1]
        margin = max(0.04, 0.02 * abs(chosen["quality"]))
        if chosen["quality"] - alternate["quality"] <= margin:
            return None, [
                "2×2 row-major 与 clockwise/snake 证据接近，拓扑有歧义；"
                "保留规则网格。"
            ]

    labels = {
        row_mapping: "row-major [TL,TR,BL,BR]",
        snake_mapping: "clockwise/snake [TL,TR,BR,BL]",
    }
    topology = labels[chosen["mapping"]]
    logs = [
        f"2×2 选择拓扑：{topology}，联合质量={chosen['quality']:.3f}，"
        f"闭环误差 {chosen['closure']:.2f}px；邻边拼接估计，非物理位置确认",
    ]
    for label, _source, _target, _axis, edge in chosen["edge_specs"]:
        logs.append(
            f"接缝 {label}：位移 ({edge[0]}, {edge[1]})，"
            f"quality={edge[2]:.3f}"
        )
    return chosen["positions"], logs


def align_two_row_tiles(prepared):
    """Jointly align a row-major two-row grid with a vertical-state DP."""
    tile_count = len(prepared)
    if tile_count == 0:
        return [], []
    if tile_count == 1:
        return [(0, 0)], []

    cols = (tile_count + 1) // 2
    bottom_count = tile_count - cols
    cell_count = max(0, bottom_count - 1)
    cache = {}
    top_edges = {}
    bottom_edges = {}
    vertical_edges = {}

    def get_top(column):
        edge = _fetch_grid_candidates(
            cache, prepared, column, column + 1, "horizontal"
        )
        top_edges[column] = edge
        return edge

    def get_bottom(column):
        source = cols + column
        edge = _fetch_grid_candidates(
            cache, prepared, source, source + 1, "horizontal"
        )
        bottom_edges[column] = edge
        return edge

    def get_vertical(column):
        source = column
        target = cols + column
        edge = _fetch_grid_candidates(cache, prepared, source, target, "vertical")
        vertical_edges[column] = edge
        return edge

    # Fetch edges in cell order so the match order remains top, left, right,
    # bottom.  A cached right rung becomes the next cell's left state.
    for column in range(cell_count):
        get_top(column)
        get_vertical(column)
        get_vertical(column + 1)
        get_bottom(column)
    if not vertical_edges:
        get_vertical(0)
    if bottom_count < cols:
        get_top(cell_count)

    missing = []
    if not vertical_edges.get(0):
        missing.append("竖边 1")
    for column in range(cell_count):
        for label, edges in (
            (f"上排 {column + 1}-{column + 2}", top_edges.get(column)),
            (f"下排 {column + 1}-{column + 2}", bottom_edges.get(column)),
            (f"竖边 {column + 2}", vertical_edges.get(column + 1)),
        ):
            if not edges:
                missing.append(label)
    if bottom_count < cols and not top_edges.get(cell_count):
        missing.append(f"上排 {cell_count + 1}-{cell_count + 2}")
    if missing:
        return None, [
            "两行网格缺少可靠匹配证据：" + "、".join(missing)
            + "；保留规则网格，请检查图片顺序或纹理。"
        ]

    tolerance = _grid_tolerance(prepared)
    initial = vertical_edges[0]
    layers = [{
        index: {"score": float(edge[2]), "previous": None}
        for index, edge in enumerate(initial)
    }]
    for column in range(cell_count):
        next_layer = {}
        for left_index, state in layers[-1].items():
            left = vertical_edges[column][left_index]
            for top in top_edges[column]:
                for right_index, right in enumerate(vertical_edges[column + 1]):
                    for bottom in bottom_edges[column]:
                        closure_x = top[0] + right[0] - left[0] - bottom[0]
                        closure_y = top[1] + right[1] - left[1] - bottom[1]
                        closure = math.hypot(closure_x, closure_y)
                        if closure > tolerance:
                            continue
                        score = (
                            state["score"]
                            + float(top[2])
                            + float(right[2])
                            + float(bottom[2])
                        )
                        previous = next_layer.get(right_index)
                        if previous is None or score > previous["score"]:
                            next_layer[right_index] = {
                                "score": score,
                                "previous": left_index,
                                "top": top,
                                "right": right,
                                "bottom": bottom,
                                "closure": closure,
                            }
        if not next_layer:
            return None, [
                f"第 {column + 1} 个 2×2 单元没有满足闭环容差 {tolerance:.2f}px 的一致路径；"
                "保留规则网格。"
            ]
        layers.append(next_layer)

    tail = None
    if bottom_count < cols:
        tail = max(top_edges[cell_count], key=lambda edge: float(edge[2]))
    final_state = max(
        layers[-1].items(),
        key=lambda item: item[1]["score"] + (float(tail[2]) if tail else 0.0),
    )
    state_index = final_state[0]
    selected = {}
    for column in range(cell_count - 1, -1, -1):
        state = layers[column + 1][state_index]
        selected[("top", column)] = state["top"]
        selected[("vertical", column + 1)] = state["right"]
        selected[("bottom", column)] = state["bottom"]
        state_index = state["previous"]
    selected[("vertical", 0)] = vertical_edges[0][state_index]
    if tail is not None:
        selected[("top", cell_count)] = tail

    edge_specs = []
    for column in range(cols - 1):
        edge_specs.append((
            f"上排 {column + 1}-{column + 2}",
            column,
            column + 1,
            "horizontal",
            selected[("top", column)],
        ))
    for column in range(bottom_count - 1):
        edge_specs.append((
            f"下排 {column + 1}-{column + 2}",
            cols + column,
            cols + column + 1,
            "horizontal",
            selected[("bottom", column)],
        ))
    for column in range(bottom_count):
        edge_specs.append((
            f"竖边 {column + 1}",
            column,
            cols + column,
            "vertical",
            selected[("vertical", column)],
        ))
    positions, validation_error = _joint_least_squares(prepared, edge_specs)
    if positions is None:
        return None, [f"两行网格联合位置验证失败：{validation_error}，保留规则网格。"]

    path_score = final_state[1]["score"] + (float(tail[2]) if tail else 0.0)
    logs = [f"两行网格联合匹配：{cols}列，路径质量={path_score:.3f}"]
    for column in range(cell_count):
        top = selected[("top", column)]
        left = selected[("vertical", column)]
        right = selected[("vertical", column + 1)]
        bottom = selected[("bottom", column)]
        residual = math.hypot(
            top[0] + right[0] - left[0] - bottom[0],
            top[1] + right[1] - left[1] - bottom[1],
        )
        logs.append(f"单元 {column + 1}：闭环误差 {residual:.2f}px")
    for label, _source, _target, _axis, edge in edge_specs:
        logs.append(f"接缝 {label}：位移 ({edge[0]}, {edge[1]})，质量={edge[2]:.3f}")
    return positions, logs

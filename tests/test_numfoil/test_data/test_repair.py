import numpy as np

from numfoil.data.repair import repair_negative_thickness_points


def _build_points_from_thickness(x: np.ndarray, thickness: np.ndarray):
    upper = np.stack([x, 0.5 * thickness], axis=-1)
    lower = np.stack([x, -0.5 * thickness], axis=-1)
    return upper, lower


def test_repair_does_not_modify_valid_closed_edges_for_zero_target():
    """A valid closed-edge profile should remain unchanged when no repair is needed.

    If this fails, the repair routine is likely introducing correction even when
    all thickness values already satisfy the target, indicating an overly
    aggressive baseline correction or endpoint handling bug.
    """
    x = np.linspace(0.0, 1.0, 101)
    thickness = np.full_like(x, 0.05)
    thickness[0] = 0.0
    thickness[-1] = 0.0
    upper, lower = _build_points_from_thickness(x, thickness)

    repaired_upper, repaired_lower = repair_negative_thickness_points(
        upper,
        lower,
        min_thickness=0.0,
        weighting="symmetric",
        locality_sigma=0.3,
    )

    repaired_thickness = repaired_upper[:, 1] - repaired_lower[:, 1]
    # Exact match is expected: there was no deficit to fix.
    np.testing.assert_allclose(repaired_thickness, thickness, atol=1e-12)


def test_symmetric_repair_spreads_over_broader_region_for_smoothness():
    """Symmetric mode should distribute correction over many points.

    If this fails low, the correction is too localized (bump-prone). If it
    fails high in future variants, the correction may be spreading too globally.
    """
    x = np.linspace(0.0, 1.0, 101)
    thickness = np.full_like(x, 0.03)
    thickness[50] = -0.01
    upper, lower = _build_points_from_thickness(x, thickness)

    repaired_upper, repaired_lower = repair_negative_thickness_points(
        upper,
        lower,
        min_thickness=0.0,
        weighting="symmetric",
        locality_sigma=0.3,
    )

    delta = (repaired_upper[:, 1] - repaired_lower[:, 1]) - thickness
    changed_points = np.count_nonzero(np.abs(delta) > 1e-8)
    # Wide support helps avoid sharp local kinks.
    assert changed_points >= 40


def test_repair_is_centered_on_near_edge_violation_location():
    """Peak correction should align with the worst-thickness index.

    If this fails, the weighting center is drifting away from the actual
    violation location, which can distort nearby geometry.
    """
    x = np.linspace(0.0, 1.0, 101)
    thickness = np.full_like(x, 0.03)
    violation_idx = 3
    thickness[violation_idx] = -0.01
    upper, lower = _build_points_from_thickness(x, thickness)

    repaired_upper, repaired_lower = repair_negative_thickness_points(
        upper,
        lower,
        min_thickness=0.0,
        weighting="symmetric",
        locality_sigma=0.3,
    )

    delta = (repaired_upper[:, 1] - repaired_lower[:, 1]) - thickness
    peak_idx = int(np.argmax(delta))
    # The largest opening should occur exactly where thickness is minimal.
    assert peak_idx == violation_idx


def test_repair_preserves_endpoint_contact_for_zero_target():
    """Leading-edge value is preserved while interior deficits are repaired.

    If this fails, LE anchoring was broken or interior correction leaked into
    the endpoint.
    """
    x = np.linspace(0.0, 1.0, 101)
    thickness = np.full_like(x, 0.03)
    thickness[10] = -0.01
    upper, lower = _build_points_from_thickness(x, thickness)

    repaired_upper, repaired_lower = repair_negative_thickness_points(
        upper,
        lower,
        min_thickness=0.0,
        weighting="edge_anchored",
        locality_sigma=0.3,
        edge_power=1.0,
    )

    repaired_thickness = repaired_upper[:, 1] - repaired_lower[:, 1]
    # LE is always preserved exactly.
    assert np.isclose(repaired_thickness[0], thickness[0], atol=1e-12)
    assert np.all(repaired_thickness[1:] >= -1e-12)


def test_edge_anchored_keeps_trailing_edge_unchanged():
    """Edge-anchored mode should preserve existing trailing-edge thickness.

    If this fails, TE tapering is not correctly zeroing influence at the
    trailing-edge point.
    """
    x = np.linspace(0.0, 1.0, 101)
    thickness = np.full_like(x, 0.03)
    thickness[70] = -0.01
    thickness[-1] = 0.004
    upper, lower = _build_points_from_thickness(x, thickness)

    repaired_upper, repaired_lower = repair_negative_thickness_points(
        upper,
        lower,
        min_thickness=0.0,
        weighting="edge_anchored",
        locality_sigma=0.3,
        edge_power=1.0,
    )

    repaired_thickness = repaired_upper[:, 1] - repaired_lower[:, 1]
    # TE value is a strict anchor in edge_anchored mode.
    assert np.isclose(repaired_thickness[-1], thickness[-1], atol=1e-12)


def test_repair_meets_minimum_thickness_at_critical_point():
    """Repair must lift the critical point to at least the requested minimum.

    If this fails, the computed correction magnitude is insufficient or numeric
    masking around the critical point is suppressing the applied delta.
    """
    x = np.linspace(0.0, 1.0, 101)
    thickness = np.full_like(x, 0.01)
    thickness[40] = -0.02
    target_min = 0.005
    upper, lower = _build_points_from_thickness(x, thickness)

    repaired_upper, repaired_lower = repair_negative_thickness_points(
        upper,
        lower,
        min_thickness=target_min,
        weighting="symmetric",
        locality_sigma=0.3,
    )

    repaired_thickness = repaired_upper[:, 1] - repaired_lower[:, 1]
    # Core contract: the worst point is repaired up to the target threshold.
    assert repaired_thickness[40] >= (target_min - 1e-12)

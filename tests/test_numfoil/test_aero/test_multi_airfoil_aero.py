import numpy as np

from numfoil.aero.aeroresults import AeroResults
from numfoil.aero.neuralfoil.neuralfoil_wrapper import NeuralFoil
from numfoil.aero.xfoil.xfoil import XFoil


class DummyAirfoil:
    """Simple airfoil stub for batch API tests."""

    def __init__(self, name: str, offset: float = 0.0):
        self.name = name
        self.parameters = np.linspace(0.01, 0.18, 18) + offset
        self.points = np.array(
            [
                [1.0, 0.0],
                [0.5, 0.08],
                [0.0, 0.0],
                [0.5, -0.08],
                [1.0, 0.0],
            ],
            dtype=float,
        )


def _single_result(label: str, cl_shift: float = 0.0) -> AeroResults:
    """Build a deterministic single-airfoil result for tests."""
    result = AeroResults(label, Mach=0.0, source="test")
    alpha = np.array([0.0, 4.0, 8.0], dtype=float)

    for re_value, delta in ((1e5, 0.00), (2e5, 0.05)):
        result.polar._add(
            re_value,
            {
                "alpha": alpha,
                "CL": np.array([0.10, 0.80, 0.50], dtype=float)
                + cl_shift
                + delta,
                "CD": np.array([0.020, 0.030, 0.050], dtype=float),
                "CM": np.array([-0.02, -0.03, -0.04], dtype=float),
            },
        )

    return result


def test_aeroresults_multi_airfoil_indexing_and_metrics():
    first = _single_result("A", cl_shift=0.0)
    second = _single_result("B", cl_shift=0.2)

    multi = AeroResults.from_airfoils({"A": first, "B": second}, label="batch")

    assert multi.is_multi_airfoil
    assert multi["A"] is first
    assert multi[1] is second
    assert multi(airfoil="A").label == "A"

    cl_max = multi.CL_max
    assert set(cl_max.keys()) == {"A", "B"}
    assert set(cl_max["A"].keys()) == {1e5, 2e5}

    df = multi.metric("CL_max", as_dataframe=True)
    assert {"airfoil", "Re", "CL_max"}.issubset(df.columns)
    assert len(df) == 4
    assert set(df["airfoil"]) == {"A", "B"}
    assert set(df["Re"]) == {1e5, 2e5}


def test_aeroresults_to_polar_tensor_single_and_multi():
    first = _single_result("A", cl_shift=0.0)
    second = _single_result("B", cl_shift=0.2)
    multi = AeroResults.from_airfoils({"A": first, "B": second}, label="batch")

    single_tensor = first.to_polar_tensor(columns=["CL", "CD"])
    assert single_tensor["airfoils"] == ["A"]
    assert single_tensor["columns"] == ["CL", "CD"]
    assert tuple(single_tensor["values"].shape) == (1, 2, 3, 2)
    np.testing.assert_allclose(single_tensor["Re"], np.array([1e5, 2e5]))
    np.testing.assert_allclose(single_tensor["alpha"], np.array([0.0, 4.0, 8.0]))

    cl_re_1e5 = first.polar(1e5)["CL"]
    np.testing.assert_allclose(single_tensor["values"][0, 0, :, 0], cl_re_1e5)

    multi_tensor = multi.to_polar_tensor(columns=["CL"])
    assert multi_tensor["airfoils"] == ["A", "B"]
    assert multi_tensor["columns"] == ["CL"]
    assert tuple(multi_tensor["values"].shape) == (2, 2, 3, 1)

    cl_a_re_2e5 = multi["A"].polar(2e5)["CL"]
    cl_b_re_2e5 = multi["B"].polar(2e5)["CL"]
    np.testing.assert_allclose(multi_tensor["values"][0, 1, :, 0], cl_a_re_2e5)
    np.testing.assert_allclose(multi_tensor["values"][1, 1, :, 0], cl_b_re_2e5)


def test_aeroresults_to_polar_tensor_requires_shared_grid():
    first = AeroResults("A", Mach=0.0, source="test")
    second = AeroResults("B", Mach=0.0, source="test")

    first.polar._add(
        1e5,
        {
            "alpha": np.array([0.0, 5.0], dtype=float),
            "CL": np.array([0.2, 0.6], dtype=float),
            "CD": np.array([0.02, 0.04], dtype=float),
        },
    )
    second.polar._add(
        1e5,
        {
            "alpha": np.array([0.0, 4.0, 8.0], dtype=float),
            "CL": np.array([0.2, 0.5, 0.7], dtype=float),
            "CD": np.array([0.02, 0.03, 0.05], dtype=float),
        },
    )

    multi = AeroResults.from_airfoils({"A": first, "B": second}, label="batch")

    try:
        multi.to_polar_tensor(columns=["CL"])
    except ValueError as exc:
        assert "alpha grids differ" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched alpha grids")


def test_reynolds_property_includes_cp_and_dump_when_polar_missing():
    result = AeroResults("cpdump", Mach=0.0, source="test")
    result.cp._add(1.5, 1e5, {"x": np.array([0.0, 1.0]), "Cp": np.array([0.1, -0.1])})
    result.dump._add(
        2.0,
        2e5,
        {
            "x": np.array([0.0, 1.0]),
            "Dstar": np.array([0.01, 0.02]),
            "Theta": np.array([0.005, 0.01]),
        },
    )

    assert result.reynolds == [1e5, 2e5]


def test_xfoil_analyze_loops_over_airfoil_batch(monkeypatch):
    solver = XFoil()
    calls = []

    def fake_analyze_single(
        self,
        airfoil,
        alphas,
        reynolds,
        Mach,
        flap,
        label,
    ):
        calls.append((airfoil, label))
        use_label = label or getattr(airfoil, "name", "airfoil")
        return _single_result(use_label, cl_shift=0.1 * len(calls))

    monkeypatch.setattr(XFoil, "_analyze_single", fake_analyze_single)

    out = solver.analyze(
        airfoil=[DummyAirfoil("foil_A"), DummyAirfoil("foil_B")],
        alphas=np.array([0.0, 4.0]),
        reynolds=np.array([1e5]),
        Mach=0.0,
    )

    assert len(calls) == 2
    assert out.is_multi_airfoil
    assert out.airfoil_labels == ["foil_A", "foil_B"]


def test_neuralfoil_analyze_mega_batches_all_airfoils(monkeypatch):
    solver = NeuralFoil.__new__(NeuralFoil)
    solver.model_size = "xxxlarge"
    solver.bl_x_points = np.array([0.25, 0.75], dtype=float)

    call_count = {"n": 0}

    def fake_eval_from_kulfan(
        self,
        kulfan,
        alpha,
        re_value,
        n_crit=9.0,
        xtr_upper=1.0,
        xtr_lower=1.0,
    ):
        call_count["n"] += 1

        alpha_vec = np.asarray(alpha, dtype=float).reshape(-1)
        re_vec = np.asarray(re_value, dtype=float).reshape(-1)
        n_cases = len(alpha_vec)

        assert len(re_vec) == n_cases
        assert np.asarray(kulfan["upper_weights"]).shape == (8, n_cases)
        assert np.asarray(kulfan["lower_weights"]).shape == (8, n_cases)

        out = {
            "analysis_confidence": np.full(n_cases, 0.95),
            "CL": 0.2 + 0.03 * alpha_vec + 1e-7 * re_vec,
            "CD": np.full(n_cases, 0.01),
            "CM": np.full(n_cases, -0.02),
            "Top_Xtr": np.full(n_cases, 0.7),
            "Bot_Xtr": np.full(n_cases, 0.8),
        }

        for i in range(self.n_bl_points):
            out[f"upper_bl_theta_{i}"] = np.full(n_cases, 1e-3 + 1e-4 * i)
            out[f"upper_bl_H_{i}"] = np.full(n_cases, 2.2 + 0.1 * i)
            out[f"upper_bl_ue/vinf_{i}"] = np.full(n_cases, 1.1 + 0.1 * i)
            out[f"lower_bl_theta_{i}"] = np.full(n_cases, 1.2e-3 + 1e-4 * i)
            out[f"lower_bl_H_{i}"] = np.full(n_cases, 2.3 + 0.1 * i)
            out[f"lower_bl_ue/vinf_{i}"] = np.full(n_cases, 1.0 + 0.1 * i)

        return out

    monkeypatch.setattr(NeuralFoil, "_evaluate_from_kulfan", fake_eval_from_kulfan)

    out = solver.analyze(
        airfoil=[DummyAirfoil("A", 0.00), DummyAirfoil("B", 0.01)],
        alphas=np.array([0.0, 5.0]),
        reynolds=np.array([1e5, 2e5]),
    )

    assert call_count["n"] == 1
    assert out.is_multi_airfoil
    assert out.airfoil_labels == ["A", "B"]

    polar_df = out.to_dataframe()
    assert len(polar_df) == 8
    assert set(polar_df["airfoil"]) == {"A", "B"}

    clmax_df = out.metric("CL_max", as_dataframe=True)
    assert len(clmax_df) == 4
    assert set(clmax_df["airfoil"]) == {"A", "B"}
    assert set(clmax_df["Re"]) == {1e5, 2e5}

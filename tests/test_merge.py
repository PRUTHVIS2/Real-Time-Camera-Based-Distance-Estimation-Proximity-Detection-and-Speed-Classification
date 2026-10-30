"""Unit tests for rider and two-wheeler merging, memory suppression, and edge truncation."""

import pytest

from vision.merge import (
    RiderMerger,
    compute_containment,
    is_box_truncated,
    is_rider_of_bike,
    merge_riders,
)
from vision.types import Detection, TrackedObject


def test_p2_a1_person_well_inside_motorcycle():
    """P2-A1: One person box well inside a motorcycle box -> 1 two_wheeler, rider_count 1, bike ID."""
    bike_det = Detection(xyxy=(100.0, 100.0, 300.0, 300.0), conf=0.9, cls="motorcycle", track_id=10)
    person_det = Detection(xyxy=(150.0, 120.0, 250.0, 260.0), conf=0.85, cls="person", track_id=1)

    merger = RiderMerger(min_containment=0.5)
    objs = merger.merge([bike_det, person_det], t_capture=1.0)

    assert len(objs) == 1
    obj = objs[0]
    assert obj.cls == "two_wheeler"
    assert obj.track_id == 10
    assert obj.rider_count == 1
    assert obj.box_xyxy == (100.0, 100.0, 300.0, 300.0)


def test_p2_a2_person_standing_beside_motorcycle():
    """P2-A2: Person standing beside motorcycle (small overlap) -> 1 person + 1 two_wheeler (rider_count 0)."""
    bike_det = Detection(xyxy=(200.0, 100.0, 400.0, 300.0), conf=0.9, cls="motorcycle", track_id=10)
    # Person only overlaps by 20 px horizontally (x=180 to 200 is outside bike, x=200 to 220 is inside)
    person_det = Detection(xyxy=(100.0, 100.0, 220.0, 300.0), conf=0.85, cls="person", track_id=1)

    merger = RiderMerger(min_containment=0.5)
    objs = merger.merge([bike_det, person_det], t_capture=1.0)

    assert len(objs) == 2
    classes = {o.cls: o for o in objs}
    assert "two_wheeler" in classes
    assert "person" in classes
    assert classes["two_wheeler"].rider_count == 0
    assert classes["two_wheeler"].track_id == 10
    assert classes["person"].track_id == 1


def test_p2_a3_two_persons_on_one_motorcycle_pillion():
    """P2-A3: Two persons on one motorcycle (pillion) -> 1 two_wheeler, rider_count 2."""
    bike_det = Detection(xyxy=(100.0, 100.0, 400.0, 300.0), conf=0.9, cls="motorcycle", track_id=10)
    p1 = Detection(xyxy=(140.0, 120.0, 220.0, 260.0), conf=0.85, cls="person", track_id=1)
    p2 = Detection(xyxy=(240.0, 120.0, 320.0, 260.0), conf=0.80, cls="person", track_id=2)

    merger = RiderMerger(min_containment=0.5)
    objs = merger.merge([bike_det, p1, p2], t_capture=1.0)

    assert len(objs) == 1
    assert objs[0].cls == "two_wheeler"
    assert objs[0].track_id == 10
    assert objs[0].rider_count == 2


def test_p2_a4_person_on_bicycle():
    """P2-A4: Person on a bicycle -> merged as two_wheeler."""
    bike_det = Detection(xyxy=(100.0, 100.0, 300.0, 300.0), conf=0.9, cls="bicycle", track_id=20)
    person_det = Detection(xyxy=(150.0, 120.0, 250.0, 260.0), conf=0.85, cls="person", track_id=1)

    merger = RiderMerger(min_containment=0.5)
    objs = merger.merge([bike_det, person_det], t_capture=1.0)

    assert len(objs) == 1
    assert objs[0].cls == "two_wheeler"
    assert objs[0].track_id == 20
    assert objs[0].rider_count == 1


def test_p2_a5_person_overlapping_two_bikes_assigned_to_larger_containment():
    """P2-A5: Person overlapping two bikes, more contained in bike B -> assigned to B only."""
    # Bike A: x in [100, 220]
    bike_a = Detection(xyxy=(100.0, 100.0, 220.0, 300.0), conf=0.85, cls="motorcycle", track_id=10)
    # Bike B: x in [150, 350]
    bike_b = Detection(xyxy=(150.0, 100.0, 350.0, 300.0), conf=0.90, cls="motorcycle", track_id=20)
    # Person: x in [180, 280] (width 100).
    # Overlap with A: [180, 220] -> width 40 (40%)
    # Overlap with B: [180, 280] -> width 100 (100%)
    person = Detection(xyxy=(180.0, 120.0, 280.0, 260.0), conf=0.88, cls="person", track_id=1)

    merger = RiderMerger(min_containment=0.5)
    objs = merger.merge([bike_a, bike_b, person], t_capture=1.0)

    assert len(objs) == 2
    by_id = {o.track_id: o for o in objs}
    assert by_id[10].rider_count == 0  # Bike A
    assert by_id[20].rider_count == 1  # Bike B


def test_p2_a6_rider_memory_suppression():
    """P2-A6: 5 frames merged, then 3 frames with no bike -> stays suppressed; reappears after threshold."""
    bike_det = Detection(xyxy=(100.0, 100.0, 300.0, 300.0), conf=0.9, cls="motorcycle", track_id=10)
    person_det = Detection(xyxy=(150.0, 120.0, 250.0, 260.0), conf=0.85, cls="person", track_id=1)

    merger = RiderMerger(min_containment=0.5, rider_memory_frames=3)

    # 5 frames merged
    for f in range(5):
        objs = merger.merge([bike_det, person_det], t_capture=float(f))
        assert len(objs) == 1
        assert objs[0].cls == "two_wheeler"
        assert objs[0].track_id == 10

    # 3 frames with bike missing but person still detected
    for f in range(5, 8):
        objs = merger.merge([person_det], t_capture=float(f))
        # Person should remain suppressed during the 3 flicker frames!
        assert len(objs) == 0, f"Expected person to be suppressed on frame {f}"

    # 4th frame without bike: rider_memory_frames (3) exceeded, person should reappear as pedestrian!
    objs = merger.merge([person_det], t_capture=8.0)
    assert len(objs) == 1
    assert objs[0].cls == "person"
    assert objs[0].track_id == 1


def test_p2_a7_box_touching_frame_edge_marked_truncated():
    """P2-A7: Box touching frame edge -> truncated = True."""
    f_size = (1280, 720)
    # Touching left edge (x1 <= 3)
    assert is_box_truncated((2.0, 100.0, 200.0, 300.0), frame_size=f_size, margin_px=3)
    # Touching bottom edge (y2 >= 720 - 3 = 717)
    assert is_box_truncated((100.0, 100.0, 200.0, 718.0), frame_size=f_size, margin_px=3)
    # Well inside
    assert not is_box_truncated((50.0, 50.0, 300.0, 400.0), frame_size=f_size, margin_px=3)

    # In merger output
    det_truncated = Detection(xyxy=(1.0, 50.0, 200.0, 400.0), conf=0.9, cls="car", track_id=5)
    merger = RiderMerger(default_frame_size=f_size, truncation_margin_px=3)
    res = merger.merge([det_truncated], t_capture=1.0)
    assert len(res) == 1
    assert res[0].truncated is True


def test_p2_a8_empty_input_produces_empty_output():
    """P2-A8: Empty input -> empty output without error."""
    merger = RiderMerger()
    assert merger.merge([], t_capture=1.0) == []

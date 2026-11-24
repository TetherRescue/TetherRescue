"""육지 출항 → GPS 자동 표기 → 항해 시나리오 검사."""
import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from sim.world.coast_map import SHORE_X
from sim.world.sortie import RescueSortie, SortieSettings, SortieStage


def test_boat_starts_on_land_then_auto_marks_victim_gps():
    sortie = RescueSortie(settings=SortieSettings(launch_delay=.2, launch_hold=.1, victim_range=30.))
    assert sortie.stage == SortieStage.STOWED
    assert sortie.world.data.xpos[sortie.robot.body_id, 0] > SHORE_X
    assert sortie.gps_fix is None

    while sortie.stage == SortieStage.STOWED:
        sortie.step()
    assert sortie.stage == SortieStage.LAUNCHING
    assert sortie.world.data.xpos[sortie.robot.body_id, 0] < SHORE_X

    while sortie.stage == SortieStage.LAUNCHING:
        sortie.step()
    assert sortie.stage == SortieStage.TRANSIT
    victim = sortie.world.data.xpos[sortie.human.body_id, :2]
    np.testing.assert_allclose(sortie.gps_fix, victim, atol=1e-6)
    assert sortie.controller is not None


def test_sortie_reaches_short_range_victim_gps():
    # 긴 200m 대신 짧은 거리로 도착까지 검사한다.
    from sim.robot.phase_manager import ApproachSettings
    approach = ApproachSettings(max_speed=2.0, distance_gain=.4, capture_radius=2.0,
                                position_tolerance=4.0, reapproach_radius=6.0,
                                arrival_hold_seconds=.5)
    sortie = RescueSortie(settings=SortieSettings(
        launch_delay=.05, launch_hold=.05, victim_range=25., approach=approach))
    for _ in range(80000):
        sortie.step()
        if sortie.arrived:
            break
        assert np.isfinite(sortie.world.data.qpos).all()
    assert sortie.arrived
    assert sortie.readings()['distance_to_gps'] <= 4.0 + 1e-6

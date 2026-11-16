"""배 ↔ 사람·구명환 실제 충돌 시험 (담당: 가영, 통합 충돌 확인).

config.Contacts 표에서 배·사람·구명환은 서로 부딪히게 되어 있다. test_world는 비트 설정만 보고,
여기서는 잔잔한 물 위에서 실제로 부딪혔을 때 물리가 멀쩡한지 확인한다.
장면은 sim/scenarios.py의 ContactDemo이며 화면(python -m sim.view_contact)과 같다.

구조 접근(RescueApproach)은 0.1m 안에서 미리 멈추므로, 이 시험은 그 안전장치가 실패했을 때의
최악 상황을 대신한다.

2026-11-16 측정 (사람은 물에 5초 자리 잡은 자세, 배와 최소 간격 0.15m에서 출발):
  1 m/s 충돌(발·옆·머리 쪽): 겹침 최대 4.0mm, 사람 속도 변화 최대 0.18 m/s (완전 탄성 한계 0.43 m/s)
  최대 추력으로 5초 밀기(발·옆): 겹침 최대 5.1mm, 사람 약 1m 밀려남. 옆에서 밀면 배가 회전해 사람 둘레를 돈다.
  모든 경우 경고·NaN 없음 → 사람·구명환 solref(.006, 1)·solimp 설정을 바꾸지 않아도 된다.
"""

import pytest

pytest.importorskip("mujoco")

from sim.scenarios import ContactDemo  # noqa: E402

MAX_PENETRATION = 0.01        # m. 충돌 겹침 허용 (측정 최대 4mm)
MAX_PUSH_PENETRATION = 0.015  # m. 최대 추력으로 계속 밀 때 (측정 최대 5.1mm)
HEADINGS = ContactDemo.HEADINGS


def _run(demo, seconds):
    for _ in range(round(seconds / demo.world.model.opt.timestep)):
        demo.step()
    return demo.readings()


def test_scene_starts_apart_at_requested_gap():
    demo = ContactDemo(HEADINGS["side"], gap=0.2)
    assert demo.gap() == pytest.approx(0.2, abs=0.01)
    assert not demo.boat_contacts()


def test_reset_restores_start_and_clears_records():
    demo = ContactDemo(HEADINGS["feet"])
    start = demo.world.data.qpos.copy()
    _run(demo, 1.0)
    assert demo.hit
    demo.reset()
    assert (demo.world.data.qpos == start).all()
    assert demo.world.data.time == 0
    assert not demo.hit and demo.max_penetration == 0


@pytest.mark.parametrize("heading_deg", HEADINGS.values(), ids=HEADINGS.keys())
def test_boat_impact_does_not_pass_through_or_add_energy(heading_deg):
    # 1 m/s: 구조 접근 목표 속도(0.2~0.4 m/s)보다 빠른 최악 상황
    demo = ContactDemo(heading_deg, speed=1.0)
    record = _run(demo, 3.0)

    assert record["hit"], "배가 사람·환에 닿지 않음 — 장면 배치를 확인할 것"
    assert record["max_penetration"] < MAX_PENETRATION
    assert 0 < record["human_dv"] < record["elastic_limit"], "충돌로 에너지가 생겨남 (튕겨 나가는 속도가 너무 큼)"
    assert record["push"] > 0, "사람이 배가 미는 방향으로 밀려나지 않음"
    assert record["warnings"] == 0


@pytest.mark.parametrize("heading_deg", [HEADINGS["feet"], HEADINGS["side"]], ids=["feet", "side"])
def test_full_thrust_push_stays_stable(heading_deg):
    # 최대 추력으로 계속 밀어도 뚫고 지나가지 않고 사람을 앞으로 밀어낸다.
    # 옆(구명환)에서 밀면 배가 마찰로 회전해 사람 둘레를 닿은 채 돌아간다 — 겹침만 작으면 정상.
    demo = ContactDemo(heading_deg, speed=0.0, thrust=1.0)
    record = _run(demo, 5.0)

    assert record["hit"]
    assert record["max_penetration"] < MAX_PUSH_PENETRATION
    assert record["push"] > 0.3, "계속 미는데 사람이 거의 움직이지 않음 (끼임 의심)"
    assert record["warnings"] == 0

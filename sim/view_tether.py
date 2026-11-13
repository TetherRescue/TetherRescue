"""로봇 대신 시험용 고정점을 사용한 구명환·사람 견인 데모.

실행: python -m sim.view_tether [--headless]
물·실제 선체·연결 단계 전환·끊김은 이 독립 데모에 포함하지 않는다.
"""

import argparse
import time

import mujoco
import numpy as np

from sim import config
from sim.human import Human
from sim.lifebuoy import Lifebuoy
from sim.rope_visual import rope_curve
from sim.tether import Tether
from sim.view_human import run_headless, run_viewer
from sim.world import build_world


class DemoAnchor:
    """선미 연결점 이름만 제공하는 시험용 월드 고정점. 로봇 구현을 대체하지 않는다."""
    REQUIRES = {}

    def __init__(self, *, floor=False):
        self.floor = floor

    def build(self, spec):
        spec.worldbody.add_site(name=config.Names.TOW_POINT_SITE,
                                pos=[3.0, 0.325, 1.0], size=[0.06, 0, 0],
                                rgba=[0.2, 0.8, 1.0, 1.0])
        if self.floor:
            # 화면의 이동 기준. 충돌 없는 시각용 바닥으로 부력이나 지지력을 만들지 않는다.
            spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE,
                                    size=[8, 8, 0.1], rgba=[0.16, 0.2, 0.25, 1],
                                    contype=0, conaffinity=0)

    def bind(self, model):
        pass

    def add_forces(self, model, data, acc):
        pass


def build_demo_world():
    # 실제 연결 시점의 확정이 아니라 시험에서만 처음부터 연결하는 설정이다.
    return build_world(parts=[DemoAnchor(), Human(), Lifebuoy(attached=True), Tether(enabled=True)])


class PullComparison:
    """느슨함 2초 → 늘어난 초기 상태를 1초 정지 표시 → 놓고 4초 관찰."""
    def __init__(self):
        self.rope = Tether(enabled=True)
        self.world = build_world(
            parts=[DemoAnchor(floor=True), Human(), Lifebuoy(attached=True), self.rope], gravity=0.0,
        )
        self.clock = 0.0
        self.phase = None
        self.start_x = None
        self._set_phase("SLACK")

    def _set_phase(self, phase):
        world = self.world
        mujoco.mj_resetData(world.model, world.data)
        # 두 번째 설정으로 바꿀 때의 점프는 실험 리셋이지 로프가 만든 움직임이 아니다.
        # 현재 site의 x 좌표에서 원하는 초기 로프 길이만큼 뒤로 사람을 배치한다.
        mujoco.mj_forward(world.model, world.data)
        start, end = (world.data.site_xpos[i] for i in self.rope.site_ids)
        rest = world.model.tendon_lengthspring[self.rope.tendon_id, 1]
        length = 0.75 * rest if phase == "SLACK" else rest + 1.0
        offset = start[0] - end[0] - length
        world.data.qpos[0] += offset
        ring_adr = world.model.jnt_qposadr[world.model.joint(config.Names.LIFEBUOY_FREE).id]
        world.data.qpos[ring_adr] += offset
        world.acc.reset()
        world.acc.apply(world.model, world.data)
        mujoco.mj_forward(world.model, world.data)
        self.start_x = float(world.data.qpos[0])
        self.phase = phase

    def advance(self):
        # 시연 시간과, 일시정지 중에는 흐르지 않는 물리 시간을 구분한다.
        if self.clock >= 7.0:
            self.clock = 0.0
            self._set_phase("SLACK")
        phase = "SLACK" if self.clock < 2 else "HELD" if self.clock < 3 else "RELEASED"
        if phase != self.phase:
            if phase == "HELD":
                self._set_phase(phase)
            else:
                self.phase = phase
        world = self.world
        if phase != "HELD":
            mujoco.mj_forward(world.model, world.data)
            world.acc.reset()
            for part in world.parts:
                part.add_forces(world.model, world.data, world.acc)
            world.acc.apply(world.model, world.data)
            mujoco.mj_step(world.model, world.data)
            mujoco.mj_forward(world.model, world.data)
        self.clock += world.model.opt.timestep

    def readings(self):
        model, data = self.world.model, self.world.data
        length = float(data.ten_length[self.rope.tendon_id])
        rest = float(model.tendon_lengthspring[self.rope.tendon_id, 1])
        return dict(length=length, rest=rest, rope_length=max(length, rest),
                    slack=max(0., rest - length), extension=max(0., length - rest),
                    tension=self.rope.tension(model, data),
                    displacement=float(data.qpos[0]) - self.start_x)

    def overlay(self):
        reading = self.readings()
        titles = {"SLACK": "1 / SLACK - no pulling force",
                  "HELD": "2 / RESET TO STRETCHED SETUP - paused before release",
                  "RELEASED": "3 / RELEASED - watch the person follow"}
        return (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                "TETHER TEST (gravity OFF)\n" + titles[self.phase] + "\n\n"
                "Endpoint gap\nRope / rest length\nSlack (extra rope)\nExtension\nTension\nPerson movement\n\n"
                "Gray: slack | Orange: pulling\nRed arrow: force (scaled)\nBlue line: initial position",
                "\n\n\n"
                f"{reading['length']:.2f} m\n"
                f"{reading['rope_length']:.2f} / {reading['rest']:.2f} m\n"
                f"{reading['slack']:.2f} m\n"
                f"{reading['extension'] * 100:.1f} cm\n"
                f"{reading['tension']:.1f} N\n{reading['displacement']:+.2f} m")

    def decorate(self, scene, *, clear=True):
        reading = self.readings()
        taut = reading["tension"] > 0.01
        # 색상·두께와 화살표 크기는 표시용이며 실제 힘 계산은 바꾸지 않는다.
        model = self.world.model
        # 기존 엔진의 직선 표시는 숨기고, 표시용 중심선을 별도로 그린다.
        # alpha 변경은 표시만 바꾸며 tendon의 장력·길이 계산은 유지한다.
        model.tendon_rgba[self.rope.tendon_id, 3] = 0.0
        if clear:
            scene.ngeom = 0

        def connector(kind, width, start, end, color):
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, kind, np.zeros(3), np.zeros(3), np.eye(3).ravel(), color)
            mujoco.mjv_connector(geom, kind, width, np.asarray(start), np.asarray(end))
            scene.ngeom += 1

        anchor, ring = (self.world.data.site_xpos[i] for i in self.rope.site_ids)
        points = rope_curve(anchor, ring, reading["rest"])
        color = [1., .4, .05, 1.] if taut else [.7, .75, .8, 1.]
        for start, end in zip(points[:-1], points[1:]):
            connector(mujoco.mjtGeom.mjGEOM_CAPSULE, .012, start, end, color)
        connector(mujoco.mjtGeom.mjGEOM_CAPSULE, .012,
                  [self.start_x, -.9, .025], [self.start_x, .9, .025], [.2, .6, 1., 1.])
        if taut:
            direction = (anchor - ring) / np.linalg.norm(anchor - ring)
            length = min(.8, reading["tension"] / 150)
            connector(mujoco.mjtGeom.mjGEOM_ARROW, .035,
                      ring, ring + length * direction, [1., .12, .08, 1.])


def run_comparison(*, headless=False):
    demo = PullComparison()
    if headless:
        for _ in range(3500):
            demo.advance()
            if not np.isfinite(demo.world.data.qpos).all() or not np.isfinite(demo.world.data.qvel).all():
                raise RuntimeError("비교 시연의 위치·속도에 비정상 값이 있습니다.")
        print("느슨함 → 늘어난 설정 → 놓기: 7초 비교 시연 정상 실행")
        return
    import mujoco.viewer

    with mujoco.viewer.launch_passive(demo.world.model, demo.world.data,
                                     show_left_ui=False, show_right_ui=False) as viewer:
        viewer.opt.geomgroup[config.Groups.PHYSICS] = 1
        with viewer.lock():
            viewer.cam.lookat[:] = (1.1, 0, .7)
            viewer.cam.distance = 6.5
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -45
        budget = 0.0
        while viewer.is_running():
            started = time.monotonic()
            with viewer.lock():
                while budget < 1 / 60:
                    demo.advance()
                    budget += demo.world.model.opt.timestep
                budget -= 1 / 60
                demo.decorate(viewer.user_scn)
            viewer.set_texts([demo.overlay()])
            viewer.sync()
            time.sleep(max(0., 1 / 60 - (time.monotonic() - started)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="창 없이 비교 시연 검사")
    parser.add_argument("--pendulum", action="store_true", help="기존 중력 낙하 시연")
    args = parser.parse_args()
    if not args.pendulum:
        run_comparison(headless=args.headless)
        return
    world = build_demo_world()
    if args.headless:
        run_headless(world)
    else:
        run_viewer(world, lookat=(1, 0, -1), distance=9)


if __name__ == "__main__":
    main()

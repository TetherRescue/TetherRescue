"""지나가던 사람이 익수자를 발견하고, 배를 찾아, 물가에 밀어 띄우는 연출.

화면용 mocap이다. 배의 추력·익수자 물리는 바꾸지 않는다.
연출이 켜진 동안만 배 자세를 붙잡고, 끝나면 그 자리(물가)에서 기존 항해를 시작한다.
동작은 네 박자로 나눈다. 걸음, 발견(가리키기), 배로 달리기, 밀어 띄우기.
"""

import math

import numpy as np

from sim import config
from sim.world.human import add_visual_human, upright_quat


_LABELS = {
    'walk': '1/4  WALK along the shore',
    'spot': '2/4  SPOT the person in the water',
    'seek': '3/4  RUN to the boat',
    'push': '4/4  PUSH the boat into the water',
    'aside': 'DONE  boat is afloat',
}
_SHIRT = (1.0, .82, .08, 1)   # 발견자만 노란 옷. 몸 형상은 익수자와 같다.


def _smooth(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _mix_angle(a, b, u):
    return a + _wrap(b - a) * u


def _yaw_quat(yaw):
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def _yaw_of(quat):
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _toward(src, dst):
    return math.atan2(dst[1] - src[1], dst[0] - src[0])


def _beat(name, duration, p0, p1, yaw0, yaw1):
    return dict(name=name, duration=float(duration),
                p0=np.asarray(p0, dtype=float), p1=np.asarray(p1, dtype=float),
                yaw0=float(yaw0), yaw1=float(yaw1))


def _float_z():
    draft = config.Expect.DRAFT or 0.0
    return config.WaterParams.SURFACE_Z + config.Hull.HEIGHT / 2 - draft


def _beach_z(x, tide):
    """월미도 모래 경사 윗면 높이. 위 끝보다 육지 쪽은 해변 위 높이로 둔다."""
    cfg = config.Wolmido
    local = float(x) - cfg.WALL_FACE_X
    (top_x, top_z), (toe_x, toe_z) = cfg.BEACH_TOP, cfg.BEACH_TOE
    if local >= top_x:
        z = top_z
    elif local <= toe_x:
        z = toe_z
    else:
        z = top_z + (local - top_x) / (toe_x - top_x) * (toe_z - top_z)
    return z + tide


class _Route:
    def __init__(self, beats, boat_stow, boat_end, boat_yaw, victim_look, aside, feet_z):
        self.beats = beats
        self.boat_stow = np.asarray(boat_stow, dtype=float)
        self.boat_end = np.asarray(boat_end, dtype=float)
        self.boat_yaw = float(boat_yaw)
        self.victim_look = np.asarray(victim_look, dtype=float)
        self.aside = np.asarray(aside, dtype=float)
        self.feet_z = feet_z

    @property
    def duration(self):
        return sum(beat['duration'] for beat in self.beats)


class ShoreCall:
    """노란 조끼·빨간 모자의 발견자. 맵이 켜기 전에는 제자리에 서 있기만 한다."""

    REQUIRES = {}

    def __init__(self, route, *, playing=False):
        self.route = route
        self.playing = playing
        self.elapsed = 0.0
        self.done = False
        self._armed = False
        self._mission = None
        self._mocap = None
        self._joints = []
        self._boat_home = self._boat_quat = None
        self._ring_adr = self._ring_dof = None
        self._ring_home = self._ring_quat = None
        self._eq = None

    @staticmethod
    def pose(name, local):
        """박자 이름과 그 박자 안의 시간[s]으로 팔·다리·허리·골반 높이를 정한다."""
        local = float(local)
        if name == 'spot':
            # 팔의 0은 아래로 늘어뜨린 자세. 2.5면 바다 쪽으로 높이 든다.
            raised = _smooth(local / 0.45)
            wave = math.sin(local * 11.0)
            return dict(leg_l=.08 * math.sin(local * 3.0), leg_r=-.08 * math.sin(local * 3.0),
                        arm_l=raised * (1.15 + .7 * wave), arm_r=raised * 2.55,
                        torso=.08 * raised, pelvis=.96, bounce=.05 * math.sin(local * 8.0) * raised)
        if name == 'seek':
            swing = math.sin(local * 2.0 * math.pi * 3.6)
            return dict(leg_l=1.05 * swing, leg_r=-1.05 * swing, arm_l=-1.15 * swing, arm_r=1.15 * swing,
                        torso=.4, pelvis=.90, bounce=.08 * abs(swing))
        if name == 'push':
            swing = math.sin(local * 2.0 * math.pi * 2.1)
            return dict(leg_l=.55 + .2 * swing, leg_r=.55 - .2 * swing, arm_l=1.15, arm_r=1.15,
                        torso=.62, pelvis=.70, bounce=.015 * swing)
        if name == 'aside':
            wave = math.sin(local * 9.0)
            return dict(leg_l=.35 * math.sin(local * 7.0), leg_r=-.35 * math.sin(local * 7.0),
                        arm_l=.2, arm_r=2.3 + .35 * wave, torso=.02, pelvis=.96, bounce=.03 * abs(wave))
        swing = math.sin(local * 2.0 * math.pi * 1.45)
        return dict(leg_l=.55 * swing, leg_r=-.55 * swing, arm_l=-.7 * swing, arm_r=.7 * swing,
                    torso=.04, pelvis=.96, bounce=.035 * abs(swing))

    def extend_walk(self, extra):
        """해안을 더 걷게 한다. 0 이하면 그대로 둔다."""
        extra = float(extra)
        if not math.isfinite(extra) or extra <= 0:
            return
        for beat in self.route.beats:
            if beat['name'] == 'walk':
                beat['duration'] += extra
                return

    @property
    def duration(self):
        return self.route.duration

    def _beat(self):
        remaining = min(self.elapsed, self.duration)
        for beat in self.route.beats:
            if remaining <= beat['duration'] or beat is self.route.beats[-1]:
                span = beat['duration']
                u = 0.0 if span <= 0 else min(remaining / span, 1.0)
                return beat, u
            remaining -= beat['duration']
        beat = self.route.beats[-1]
        return beat, 1.0

    def phase_name(self):
        return self._beat()[0]['name']

    def label(self):
        return _LABELS[self.phase_name()]

    def boat_position(self):
        name, u = self.phase_name(), self._beat()[1]
        if name == 'push':
            blend = _smooth(u)
            return (1.0 - blend) * self.route.boat_stow + blend * self.route.boat_end
        if name == 'aside' or self.elapsed >= self.duration:
            return self.route.boat_end.copy()
        return self.route.boat_stow.copy()

    def _stern(self, boat):
        yaw = self.route.boat_yaw
        forward = np.array([math.cos(yaw), math.sin(yaw)])
        right = np.array([-math.sin(yaw), math.cos(yaw)])
        xy = boat[:2] - forward * (config.Hull.LENGTH / 2 + 0.32) + right * 0.28
        return np.array([xy[0], xy[1], self.route.feet_z(float(xy[0]))])

    def foot_position(self):
        beat, u = self._beat()
        blend = _smooth(u)
        if beat['name'] == 'push':
            return self._stern(self.boat_position())
        if beat['name'] == 'aside':
            start = self._stern(self.route.boat_end)
            return (1.0 - blend) * start + blend * self.route.aside
        return (1.0 - blend) * beat['p0'] + blend * beat['p1']

    def facing(self):
        beat, u = self._beat()
        blend = _smooth(u)
        if beat['name'] == 'push':
            return self.route.boat_yaw
        return _mix_angle(beat['yaw0'], beat['yaw1'], blend)

    def pelvis_position(self):
        """선 자세의 골반(몸 원점). 발은 foot_position, 익수자 발끝은 원점보다 약 0.9 m 아래."""
        beat, u = self._beat()
        stance = self.pose(beat['name'], u * beat['duration'])
        foot = self.foot_position()
        return np.array([foot[0], foot[1], foot[2] + 0.90 + stance['bounce']])

    def sample(self, time):
        """테스트용. 시각만 바꾸고 자세 숫자를 돌려준다."""
        self.elapsed = float(np.clip(time, 0.0, self.duration))
        beat, u = self._beat()
        stance = self.pose(beat['name'], u * beat['duration'])
        return dict(phase=beat['name'], feet=self.foot_position(), boat=self.boat_position().copy(),
                    facing=self.facing(), **stance)

    def build(self, spec):
        foot = self.route.beats[0]['p0']
        pelvis = np.asarray(foot, dtype=float) + np.array([0.0, 0.0, 0.90])
        add_visual_human(spec, 'shore_caller', pelvis, upright_quat(self.route.beats[0]['yaw0']), shirt=_SHIRT)

    def bind(self, model):
        body = model.body('shore_caller')
        self._mocap = int(body.mocapid[0])
        self._joints = []
        for part in ('arm_l', 'arm_r'):
            joint = model.joint(f'shore_caller_{part}').id
            self._joints.append((int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])))

    def add_forces(self, model, data, acc):
        self.write(model, data)

    def write(self, model, data):
        if self._mocap is None:
            return
        beat, u = self._beat()
        stance = self.pose(beat['name'], u * beat['duration'])
        data.mocap_pos[self._mocap] = self.pelvis_position()
        data.mocap_quat[self._mocap] = upright_quat(self.facing(), lean=stance['torso'])
        # 팔은 옆으로 뻗은 익수자 자세에서 앞으로 든다. 연출 각이 크면 한 바퀴 돌지 않게 누른다.
        angles = (math.tanh(stance['arm_l']) * 1.15, math.tanh(stance['arm_r']) * 1.15)
        for (qadr, dadr), angle in zip(self._joints, angles):
            data.qpos[qadr] = angle
            data.qvel[dadr] = 0.0

    def arm(self, world, robot, mission):
        """월미도: 지금 물가 자세를 기억하고, 배를 모래 위로 옮겨 연출을 시작한다."""
        import mujoco

        if self._armed:
            return
        model, data = world.model, world.data
        mujoco.mj_forward(model, data)
        boat = int(model.jnt_qposadr[robot.joint_id])
        self._boat_adr = boat
        self._boat_dof = int(model.jnt_dofadr[robot.joint_id])
        self._boat_home = data.qpos[boat:boat + 3].copy()
        self._boat_quat = data.qpos[boat + 3:boat + 7].copy()
        self._boat_quat /= np.linalg.norm(self._boat_quat)
        self.route.boat_end = self._boat_home.copy()
        self.route.boat_yaw = _yaw_of(self._boat_quat)
        ring = model.joint(config.Names.LIFEBUOY_FREE).id
        self._ring_adr = int(model.jnt_qposadr[ring])
        self._ring_dof = int(model.jnt_dofadr[ring])
        self._ring_home = data.qpos[self._ring_adr:self._ring_adr + 3].copy()
        self._ring_quat = data.qpos[self._ring_adr + 3:self._ring_adr + 7].copy()
        self._eq = int(model.equality(config.Names.LAUNCHER_MOUNT_EQ).id)
        data.eq_active[self._eq] = False
        self._mission = mission
        self._armed = True
        self.playing = True
        self.place(world, robot)
        self.write(model, data)
        mujoco.mj_forward(model, data)

    def place(self, world, robot):
        """배(와 실려 있는 구명환)를 연출 위치로 둔다. 사람 속도는 건드리지 않는다."""
        model, data = world.model, world.data
        pos = self.boat_position()
        adr = int(model.jnt_qposadr[robot.joint_id])
        dof = int(model.jnt_dofadr[robot.joint_id])
        quat = _yaw_quat(self.route.boat_yaw) if self._boat_quat is None else self._boat_quat
        data.qpos[adr:adr + 3] = pos
        data.qpos[adr + 3:adr + 7] = quat
        data.qvel[dof:dof + 6] = 0.0
        if self._ring_adr is not None:
            delta = pos - self._boat_home
            data.qpos[self._ring_adr:self._ring_adr + 3] = self._ring_home + delta
            data.qpos[self._ring_adr + 3:self._ring_adr + 7] = self._ring_quat
            data.qvel[self._ring_dof:self._ring_dof + 6] = 0.0
        robot.set_command(0, 0)

    def _restore(self, world):
        import mujoco

        model, data = world.model, world.data
        data.qpos[self._boat_adr:self._boat_adr + 3] = self._boat_home
        data.qpos[self._boat_adr + 3:self._boat_adr + 7] = self._boat_quat
        data.qvel[self._boat_dof:self._boat_dof + 6] = 0.0
        data.qpos[self._ring_adr:self._ring_adr + 3] = self._ring_home
        data.qpos[self._ring_adr + 3:self._ring_adr + 7] = self._ring_quat
        data.qvel[self._ring_dof:self._ring_dof + 6] = 0.0
        data.eq_active[self._eq] = True
        mujoco.mj_forward(model, data)

    def play_step(self, world, robot):
        """연출 한 스텝. 끝났으면 True."""
        import mujoco
        from sim.world.world import advance_world

        if self.done:
            return True
        if not self.playing:
            return False
        self.elapsed = min(self.duration, self.elapsed + float(world.model.opt.timestep))
        finished = self.elapsed >= self.duration - 1e-12
        self.place(world, robot)
        advance_world(world)
        if not finished:
            self.place(world, robot)
            return False
        self.elapsed = self.duration
        if self._mission is not None:
            self._restore(world)
            self._mission.approach.reset()
            self._mission.watchdog.reset(float(world.data.time), self._mission.stage.value)
            self._mission.robot.set_command(0, 0)
        else:
            self.place(world, robot)
            mujoco.mj_forward(world.model, world.data)
        self.done = True
        return True

    def draw(self, scene):
        """박자마다 다른 표식. 걸음은 없고, 발견은 빨강, 배는 노란 기둥, 밀기는 주황 막대."""
        if self.done or scene.maxgeom - scene.ngeom < 3:
            return
        import mujoco

        name, u = self.phase_name(), self._beat()[1]
        if name == 'spot' and u >= 0.42:
            head = self.pelvis_position() + np.array([0.0, 0.0, .85])
            _sphere(scene, head + np.array([0.0, 0.0, .28]), .16, (1, .15, .08, 1))
            direction = self.route.victim_look - head
            flat = direction.copy()
            flat[2] = 0.0
            length = float(np.linalg.norm(flat))
            if length > 1e-6:
                _arrow(scene, head, head + flat / length * 2.6, .06, (1, .2, .1, 1))
        elif name == 'seek':
            boat = self.boat_position()
            base = np.array([boat[0], boat[1], boat[2] + .3])
            _arrow(scene, base, base + np.array([0.0, 0.0, 2.4]), .09, (1, .9, .1, 1))
        elif name == 'push':
            boat = self.boat_position()
            yaw = self.route.boat_yaw
            forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
            center = np.array([boat[0], boat[1], boat[2] + .2])
            # 납작한 선체가 안 보이므로, 배 위에 주황 기둥을 세워 밀리는 배라는 걸 표시한다.
            _arrow(scene, center, center + np.array([0.0, 0.0, 1.6]), .08, (1, .4, .05, 1))
            _arrow(scene, center - forward * .2, center + forward * (config.Hull.LENGTH / 2 + .8),
                   .07, (1, .55, .05, 1))


def _sphere(scene, pos, radius, rgba):
    import mujoco

    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]),
                        np.asarray(pos, dtype=float), np.eye(3).ravel(), rgba)
    scene.ngeom += 1


def _arrow(scene, start, end, radius, rgba):
    import mujoco

    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                        np.eye(3).ravel(), rgba)
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, radius,
                         np.asarray(start, dtype=float), np.asarray(end, dtype=float))
    scene.ngeom += 1


def hanagae_shore_call(victim_xy):
    """하나개: 모래사장에서 바다를 보고, 해안 쪽 배를 얕은 물까지 민다."""
    from sim.world.coast_map import SHORE_X, land_boat_pose, sand_height

    stow, yaw = land_boat_pose()
    stow = np.asarray(stow, dtype=float)
    end = np.array([SHORE_X - 8.0, stow[1], _float_z()])
    surface = config.WaterParams.SURFACE_Z

    def feet_z(x):
        return sand_height(x) if x >= SHORE_X - 25.0 else surface - 0.2

    route = _script(stow, end, yaw, np.asarray(victim_xy, dtype=float), feet_z,
                    walk_from=np.array([4.0, 14.0, sand_height(4.0)]),
                    walk_to=np.array([4.0, 5.5, sand_height(4.0)]),
                    aside=np.array([5.2, 2.4, sand_height(5.2)]))
    return ShoreCall(route, playing=True)


def wolmido_shore_call(world_map, boat_xy, boat_yaw, victim_xy):
    """월미도: 산책로에서 바다를 보고, 모래 위 배를 물가 출발 지점까지 민다."""
    cfg = config.Wolmido
    tide = float(world_map.tide_drop)
    boat_xy = np.asarray(boat_xy, dtype=float)
    stow = np.array([cfg.WALL_FACE_X + 2.4, boat_xy[1],
                     _beach_z(cfg.WALL_FACE_X + 2.4, tide) + config.Hull.HEIGHT / 2 + 0.08])
    end = np.array([boat_xy[0], boat_xy[1], _float_z()])
    top = cfg.WALL_TOP_Z + tide + 0.02

    def feet_z(x):
        ground = _beach_z(x, tide)
        if ground >= config.WaterParams.SURFACE_Z - 0.05:
            return ground + 0.02
        return config.WaterParams.SURFACE_Z - 0.22

    route = _script(stow, end, boat_yaw, np.asarray(victim_xy, dtype=float), feet_z,
                    walk_from=np.array([3.2, -2.0, top]),
                    walk_to=np.array([3.2, 4.6, top]),
                    aside=np.array([stow[0] + 0.3, stow[1] + 2.2, feet_z(stow[0])]),
                    via=np.array([1.6, 9.0, feet_z(1.6)]))
    return ShoreCall(route, playing=False)


def _script(stow, end, yaw, victim_xy, feet_z, walk_from, walk_to, aside, via=None):
    victim_look = np.array([victim_xy[0], victim_xy[1], config.WaterParams.SURFACE_Z + 0.45])
    spot_yaw = _toward(walk_to, victim_xy)
    forward = np.array([math.cos(yaw), math.sin(yaw)])
    right = np.array([-math.sin(yaw), math.cos(yaw)])
    stern_xy = stow[:2] - forward * (config.Hull.LENGTH / 2 + 0.32) + right * 0.28
    stern = np.array([stern_xy[0], stern_xy[1], feet_z(float(stern_xy[0]))])
    run_yaw = _toward(walk_to if via is None else via, stern)
    approach_yaw = _toward(walk_to, via) if via is not None else run_yaw
    distance = float(np.linalg.norm(end[:2] - stow[:2]))
    push_time = float(np.clip(distance / 2.8, 3.6, 8.0))
    beats = [
        _beat('walk', 3.2, walk_from, walk_to, _toward(walk_from, walk_to), _toward(walk_from, walk_to)),
        _beat('spot', 2.8, walk_to, walk_to, _toward(walk_from, walk_to), spot_yaw),
    ]
    if via is not None:
        beats.append(_beat('seek', 1.6, walk_to, via, approach_yaw, approach_yaw))
        beats.append(_beat('seek', 1.8, via, stern, run_yaw, yaw))
    else:
        beats.append(_beat('seek', 3.0, walk_to, stern, _toward(walk_to, stern), yaw))
    beats.append(_beat('push', push_time, stern, stern, yaw, yaw))
    beats.append(_beat('aside', 1.8, stern, aside, yaw, _toward(stern, aside)))
    return _Route(beats, stow, end, yaw, victim_look, aside, feet_z)

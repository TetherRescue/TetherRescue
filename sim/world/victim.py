"""물에 빠진 사람 조립. 맵은 좌표와 배 방향만 넘긴다.

선 자세·허우적·공황·구명환이 가까우면 잦아드는 동작은 Struggle 하나다.
구명환이 없는 맵은 grasp=None 으로 두고, 발버둥만 켠다.
"""

from sim import config
from sim.world.human import Human
from sim.world.struggle import Struggle


def make_victim(xy, facing, *, grasp=None):
    """(Human, Struggle)를 만든다. parts 순서는 Human, (구명환·잡기), Struggle 이다."""
    x, y = float(xy[0]), float(xy[1])
    human = Human(position=(x, y, config.WaterParams.SURFACE_Z - .03), grip_sites=True)
    return human, Struggle(human, grasp, facing=float(facing))

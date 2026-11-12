"""스모크 테스트: 뼈대가 import 되고 모듈 모양(05 문서 2장)을 지키는지 (담당: 채연).

테스트가 0개면 pytest 가 실패(exit 5)로 끝나 CI 가 빨간불이 되므로, 구현 전에도 돌아가는 최소 테스트를 둡니다.
MuJoCo 없이 돌아갑니다.
"""
import importlib

import pytest

MODULES = ["config", "forces", "frames", "water", "robot", "human", "tether", "world", "phase_manager", "watchdog"]

# 05 문서 2장: 물·로봇·사람·줄 공통 모양
PARTS = [("water", "Water"), ("robot", "Robot"), ("human", "Human"), ("tether", "Tether")]


@pytest.mark.parametrize("name", MODULES)
def test_import(name):
    importlib.import_module(f"sim.{name}")


@pytest.mark.parametrize("module, cls", PARTS)
def test_module_shape(module, cls):
    part = getattr(importlib.import_module(f"sim.{module}"), cls)
    assert isinstance(part.REQUIRES, dict)
    for method in ("build", "bind", "add_forces"):
        assert callable(getattr(part, method, None)), f"{cls}.{method} 없음"

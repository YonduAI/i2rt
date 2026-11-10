"""Exact downstream command limits are distinct from vendor encoder tolerance."""

import numpy as np
import pytest

from i2rt.robots.get_robot import _load_joint_limits_from_xml, get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.utils import ArmType


def test_explicit_joint_limits_are_unpadded_and_clip_arm_commands() -> None:
    limits = _load_joint_limits_from_xml(ArmType.YAM.get_xml_path())[:6].copy()
    limits[3] = np.deg2rad([-85.0, 85.0])
    robot = get_yam_robot(sim=True, external_gripper=True, joint_limits=limits)
    np.testing.assert_array_equal(robot.get_robot_info()["joint_limits"], limits)
    # Exercise the real driver clipping path without constructing a CAN controller.
    driver = object.__new__(MotorChainRobot)
    driver._joint_limits = limits.copy()
    driver._gripper_index = None
    for sign in (-1.0, 1.0):
        command = np.zeros(6)
        command[3] = sign * np.pi / 2
        robot.command_joint_pos(command)
        assert robot.get_joint_pos()[3] == pytest.approx(sign * np.deg2rad(85.0))
        assert driver._clip_robot_joint_pos_command(command)[3] == pytest.approx(sign * np.deg2rad(85.0))
    np.testing.assert_array_equal(limits[3], np.deg2rad([-85.0, 85.0]))
    robot.close()


@pytest.mark.parametrize("limits", [np.zeros((5, 2)), np.zeros((6, 2)), np.full((6, 2), np.nan)])
def test_invalid_explicit_joint_limits_are_rejected(limits: np.ndarray) -> None:
    with pytest.raises(ValueError, match="joint_limits"):
        get_yam_robot(sim=True, joint_limits=limits)

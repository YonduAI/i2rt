"""Exercise the library trajectory without a motor connection."""
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from i2rt.robots import motor_chain_robot


def test_joint_move_is_dense_smooth_and_uses_deadlines(monkeypatch):
    now = [0.]
    monkeypatch.setattr(motor_chain_robot.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(motor_chain_robot.time, 'sleep', lambda dt: now.__setitem__(0, now[0] + dt))
    robot = object.__new__(motor_chain_robot.MotorChainRobot)
    robot._state_lock = threading.Lock()
    robot._joint_state = SimpleNamespace(pos=np.zeros(6))
    samples = []

    def command(position):
        samples.append((now[0], position.copy()))
        now[0] += .002  # Sending must not add 2 ms to every 10 ms interval.

    robot.command_joint_pos = command
    robot.move_joints(np.full(6, np.pi / 2), time_interval_s=5.)
    times = np.array([item[0] for item in samples])
    positions = np.array([item[1] for item in samples])
    assert len(samples) == 501
    np.testing.assert_allclose(np.diff(times), .01, atol=1e-12)
    np.testing.assert_allclose(positions[-1], np.pi / 2)
    velocity = np.diff(positions, axis=0) / .01
    assert np.max(np.abs(velocity)) < .6
    assert np.max(np.abs(velocity[[0, -1]])) < .001
    assert np.max(np.abs(np.diff(velocity, axis=0))) / .01 < .4


def test_joint_feedback_carries_gravity_from_existing_control_calculation():
    robot = object.__new__(motor_chain_robot.MotorChainRobot)
    robot.motor_chain = Mock(same_bus_device_driver=None)
    robot._motor_state_to_joint_state = lambda _: SimpleNamespace()
    robot._check_current_qpos_in_joint_limits = lambda: None
    robot._joint_state_saver = None
    commands = motor_chain_robot.JointCommands.init_all_zero(6)
    gravity = np.array([0., 19., 0., 0., 0., 0.])
    robot._update_joint_state(gravity, commands, gravity_torque=gravity)
    gravity[1] = 0.
    assert robot._joint_state.gravity_torque[1] == 19.
    robot.motor_chain.set_commands.assert_called_once()

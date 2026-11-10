"""Exercise the library trajectory without a motor connection."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from i2rt.robots import motor_chain_robot


def test_stream_feedforward_preserves_position_and_expires_without_new_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    monkeypatch.setattr(motor_chain_robot.time, "monotonic", lambda: now[0])
    robot = object.__new__(motor_chain_robot.MotorChainRobot)
    robot.motor_chain = Mock()
    robot.motor_chain.__len__ = Mock(return_value=7)
    robot._command_lock = threading.Lock()
    robot._state_lock = threading.Lock()
    robot._stream_velocity_feedforward = True
    robot._position_command_time = robot._position_command = None
    robot._position_velocity = np.zeros(7)
    robot._stream_velocity_time_constant = 0.06
    robot._gripper_index = 6
    robot._gripper_limits = np.array([0.0, 1.0])
    robot._joint_limits = None
    robot._limit_gripper_force = -1
    robot._kp, robot._kd = np.ones(7) * 80, np.ones(7) * 5
    robot.remapper = motor_chain_robot.JointMapper({}, 7)
    robot._joint_state = None
    robot._clip_motor_torque = 28.0
    robot.gravity_comp_factor = 1.0
    robot.use_coulomb_friction = False
    robot._compute_gravity_compensation = lambda _: np.zeros(7)
    robot._update_joint_state = Mock()
    robot.command_joint_pos(np.zeros(7))
    np.testing.assert_array_equal(robot._commands.vel, np.zeros(7))
    for n in range(1, 101):
        now[0] = n / 100
        robot.command_joint_pos(np.full(7, now[0]))
    np.testing.assert_allclose(robot._commands.pos, 1.0)
    np.testing.assert_allclose(robot._commands.vel[:6], 0.5, atol=1e-6)
    assert robot._commands.vel[-1] == 0.0
    now[0] += 0.005
    robot.update()
    sent = robot._update_joint_state.call_args.args[1]
    np.testing.assert_array_equal(sent.pos, np.ones(7))
    np.testing.assert_allclose(sent.vel[:6], 0.5, atol=1e-6)
    assert sent.pos[6] == 1.0
    np.testing.assert_array_equal(robot._commands.pos, np.ones(7))
    now[0] += 0.01
    robot.command_joint_pos(np.full(7, 20.0))
    assert np.max(np.abs(robot._commands.vel)) <= 2.5
    now[0] += 0.051
    robot.update()
    np.testing.assert_array_equal(robot._update_joint_state.call_args.args[1].vel, np.zeros(7))
    np.testing.assert_array_equal(robot._update_joint_state.call_args.args[1].pos[:6], np.full(6, 20.0))
    robot.command_joint_pos(np.full(7, 0.6))
    np.testing.assert_array_equal(robot._commands.vel, np.zeros(7))
    now[0] += 0.01
    robot.command_joint_pos(np.full(7, 0.61))
    assert robot._commands.vel[0] > 0
    robot.command_joint_state({"pos": np.full(7, 0.61), "vel": np.full(7, 0.2)})
    assert robot._position_command_time == now[0]
    now[0] += 0.051
    robot.update()
    np.testing.assert_array_equal(robot._update_joint_state.call_args.args[1].vel, np.zeros(7))
    np.testing.assert_allclose(robot._commands.vel, 0.2)
    robot._stream_velocity_feedforward = False
    robot.command_joint_pos(np.full(7, 0.7))
    np.testing.assert_array_equal(robot._commands.vel, np.zeros(7))


def test_joint_move_is_dense_smooth_and_uses_deadlines(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]
    monkeypatch.setattr(motor_chain_robot.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(motor_chain_robot.time, "sleep", lambda dt: now.__setitem__(0, now[0] + dt))
    robot = object.__new__(motor_chain_robot.MotorChainRobot)
    robot._state_lock = threading.Lock()
    robot._joint_state = SimpleNamespace(pos=np.zeros(6))
    samples = []

    def command(position: np.ndarray) -> None:
        samples.append((now[0], position.copy()))
        now[0] += 0.002  # Sending must not add 2 ms to every 10 ms interval.

    robot.command_joint_pos = command
    robot.move_joints(np.full(6, np.pi / 2), time_interval_s=5.0)
    times = np.array([item[0] for item in samples])
    positions = np.array([item[1] for item in samples])
    assert len(samples) == 501
    np.testing.assert_allclose(np.diff(times), 0.01, atol=1e-12)
    np.testing.assert_allclose(positions[-1], np.pi / 2)
    velocity = np.diff(positions, axis=0) / 0.01
    assert np.max(np.abs(velocity)) < 0.6
    assert np.max(np.abs(velocity[[0, -1]])) < 0.001
    assert np.max(np.abs(np.diff(velocity, axis=0))) / 0.01 < 0.4


def test_joint_feedback_carries_gravity_from_existing_control_calculation() -> None:
    robot = object.__new__(motor_chain_robot.MotorChainRobot)
    robot.motor_chain = Mock(same_bus_device_driver=None)
    robot._motor_state_to_joint_state = lambda _: SimpleNamespace()
    robot._check_current_qpos_in_joint_limits = lambda: None
    robot._joint_state_saver = None
    robot._mcap_recorder = None
    robot._mcap_lock = threading.Lock()
    commands = motor_chain_robot.JointCommands.init_all_zero(6)
    gravity = np.array([0.0, 19.0, 0.0, 0.0, 0.0, 0.0])
    robot._update_joint_state(gravity, commands, gravity_torque=gravity)
    gravity[1] = 0.0
    assert robot._joint_state.gravity_torque[1] == 19.0
    robot.motor_chain.set_commands.assert_called_once()

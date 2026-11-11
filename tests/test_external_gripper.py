"""External grippers retain payload dynamics without sharing motor ownership."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest

from i2rt.motor_drivers import dm_driver
from i2rt.robots import get_robot
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml


def test_applied_command_trace_only_advances_after_a_complete_can_cycle() -> None:
    chain = object.__new__(dm_driver.DMChainCanInterface)
    chain.channel = "FAKE"
    chain.motor_list = [(1, "DM4310"), (2, "DM4310")]
    chain.motor_offset, chain.motor_direction = np.zeros(2), np.ones(2)
    chain.motor_interface = Mock()
    commands = [dm_driver.MotorCmd(pos=0.1, vel=0.2, torque=0.0, kp=20.0, kd=1.0) for _ in range(2)]
    chain.last_applied_command = None
    chain._set_commands(commands)
    completed = chain.last_applied_command
    assert completed["position_targets_rad"] == [0.1, 0.1]
    assert completed["velocity_targets_rad_s"] == [0.2, 0.2]
    chain.motor_interface.set_control.side_effect = [Mock(), TimeoutError("lost motor reply")]
    with pytest.raises(TimeoutError):
        chain._set_commands(commands)
    assert chain.last_applied_command is completed


def test_reading_cached_states_does_not_make_stalled_can_feedback_fresh() -> None:
    import threading

    chain = object.__new__(dm_driver.DMChainCanInterface)
    chain.motor_list = [(1, "DM4310")]
    chain.motor_direction, chain.motor_offset, chain.absolute_positions = np.ones(1), np.zeros(1), np.array([0.2])
    chain.state_lock = threading.Lock()
    chain.state = [
        SimpleNamespace(id=1, error_code="0x1", velocity=0.0, torque=0.0, temperature_rotor=30.0, temperature_mos=31.0)
    ]
    chain.last_applied_command = {"timestamp_ns": 1_000_000_000}
    first, second = chain.read_states(), chain.read_states()
    assert first[0].timestamp == second[0].timestamp == 1.0
    assert first[0].pos == 0.2


def test_invalid_complete_model_never_opens_can(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    factory = Mock()
    monkeypatch.setattr(get_robot, "DMChainCanInterface", factory)
    path = tmp_path / "invalid.xml"
    path.write_text("<mujoco/>")
    with pytest.raises(ValueError, match="6 scalar joints"):
        get_robot.get_yam_robot(external_gripper=True, model_xml_path=str(path))
    with pytest.raises(ValueError, match="external gripper"):
        get_robot.get_yam_robot(model_xml_path=str(path))
    factory.assert_not_called()


def test_complete_model_used_for_gravity_without_recombining(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    chain = Mock(motor_direction=np.ones(7), motor_offset=np.zeros(7))
    chain.read_states.return_value = [SimpleNamespace(pos=0.0) for _ in range(6)]
    monkeypatch.setattr(get_robot, "DMChainCanInterface", Mock(return_value=chain))
    robot = Mock()
    monkeypatch.setattr(get_robot, "MotorChainRobot", robot)
    combine = Mock(side_effect=AssertionError("must use supplied model"))
    monkeypatch.setattr(get_robot, "combine_arm_and_gripper_xml", combine)
    path = tmp_path / "arm.xml"
    bodies = "".join(f'<body pos="0 0 .1"><joint name="j{i}"/><geom size=".01"/>' for i in range(6))
    path.write_text("<mujoco><worldbody>" + bodies + "</body>" * 6 + "</worldbody></mujoco>")
    get_robot.get_yam_robot(external_gripper=True, model_xml_path=str(path))
    assert robot.call_args.kwargs["xml_path"] == str(path)
    combine.assert_not_called()


@pytest.mark.parametrize("joint_kp", [None, [240.0, 240.0, 240.0, 120.0, 30.0, 30.0]])
@pytest.mark.parametrize("joint_kd", [None, [4.0, 5.0, 4.5, 1.25, 1.0, 1.5]])
def test_external_gripper_preserves_payload_and_owns_only_arm_motors(
    monkeypatch: pytest.MonkeyPatch, joint_kp: list[float] | None, joint_kd: list[float] | None
) -> None:
    chain = Mock(motor_direction=np.ones(7), motor_offset=np.zeros(7))
    chain.read_states.return_value = [SimpleNamespace(pos=0.0) for _ in range(6)]
    factory = Mock(return_value=chain)
    robot = Mock()
    monkeypatch.setattr(get_robot, "DMChainCanInterface", factory)
    monkeypatch.setattr(get_robot, "MotorChainRobot", robot)
    get_robot.get_yam_robot(
        channel="yam_left",
        external_gripper=True,
        joint_kp=joint_kp,
        joint_kd=joint_kd,
        gravity_comp_factor=None if joint_kp is None else 1.0,
        stream_velocity_feedforward=joint_kp is not None,
    )
    for call in factory.call_args_list:
        assert [entry[0] for entry in call.args[0]] == [1, 2, 3, 4, 5, 6]
    args = robot.call_args.kwargs
    assert "gripper_index" not in args
    assert len(args["kp"]) == len(args["kd"]) == 6
    np.testing.assert_array_equal(args["kp"], [80.0, 80.0, 80.0, 10.0, 10.0, 10.0] if joint_kp is None else joint_kp)
    np.testing.assert_array_equal(args["kd"], [5.0, 5.0, 5.0, 1.5, 1.5, 1.5] if joint_kd is None else joint_kd)
    np.testing.assert_array_equal(
        args["gravity_comp_factor"], [1.0, 1.1, 1.1, 1.2, 1.0, 1.0] if joint_kp is None else np.ones(6)
    )
    assert args["stream_velocity_feedforward"] == (joint_kp is not None)
    loaded = mujoco.MjModel.from_xml_path(args["xml_path"])
    bare = mujoco.MjModel.from_xml_path(combine_arm_and_gripper_xml(ArmType.YAM, GripperType.NO_GRIPPER))
    assert loaded.body_mass.sum() > bare.body_mass.sum()
    data = mujoco.MjData(loaded)
    mujoco.mj_inverse(loaded, data)
    assert np.all(np.isfinite(data.qfrc_inverse[:6]))


@pytest.mark.parametrize("gains", [[80.0] * 5, [80.0] * 7, [float("nan")] * 6, [501.0] * 6, [0.0] * 6])
def test_invalid_position_gains_never_open_can(monkeypatch: pytest.MonkeyPatch, gains: list[float]) -> None:
    factory = Mock()
    monkeypatch.setattr(get_robot, "DMChainCanInterface", factory)
    with pytest.raises(ValueError, match="joint_kp"):
        get_robot.get_yam_robot(joint_kp=gains)
    factory.assert_not_called()


@pytest.mark.parametrize("gains", [[1.0] * 5, [1.0] * 7, [float("nan")] * 6, [5.01] * 6, [0.0] * 6])
def test_invalid_damping_never_opens_can(monkeypatch: pytest.MonkeyPatch, gains: list[float]) -> None:
    factory = Mock()
    monkeypatch.setattr(get_robot, "DMChainCanInterface", factory)
    with pytest.raises(ValueError, match="joint_kd"):
        get_robot.get_yam_robot(joint_kd=gains)
    factory.assert_not_called()


@pytest.mark.parametrize(
    "scale",
    [
        float("nan"),
        float("inf"),
        0.0,
        -1.0,
        [1.0] * 5,
        [1.0] * 7,
        [[1.0] * 6],
        [1.0, 1.0, 1.0, 1.0, 1.0, float("nan")],
        [0.0] * 6,
    ],
)
def test_invalid_gravity_scale_never_opens_can(monkeypatch: pytest.MonkeyPatch, scale: Any) -> None:
    factory = Mock()
    monkeypatch.setattr(get_robot, "DMChainCanInterface", factory)
    with pytest.raises(ValueError, match="gravity_comp_factor"):
        get_robot.get_yam_robot(gravity_comp_factor=scale)
    factory.assert_not_called()


@pytest.mark.parametrize("external", [False, True])
def test_per_joint_gravity_factors_preserve_arm_order_and_gripper(
    monkeypatch: pytest.MonkeyPatch, external: bool
) -> None:
    chain = Mock(motor_direction=np.ones(7), motor_offset=np.zeros(7))
    chain.read_states.return_value = [SimpleNamespace(pos=0.0) for _ in range(6 if external else 7)]
    monkeypatch.setattr(get_robot, "DMChainCanInterface", Mock(return_value=chain))
    robot = Mock()
    monkeypatch.setattr(get_robot, "MotorChainRobot", robot)
    factors = [1.0, 1.2, 1.0, 1.1, 1.0, 1.0]
    get_robot.get_yam_robot(external_gripper=external, gravity_comp_factor=factors)
    np.testing.assert_array_equal(
        robot.call_args.kwargs["gravity_comp_factor"], factors if external else factors + [1.0]
    )


def test_arm_socket_filters_out_external_gripper_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    interface = Mock()
    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", Mock(return_value=interface))
    monkeypatch.setattr(dm_driver.DMChainCanInterface, "_motor_on", lambda self: setattr(self, "state", []))
    dm_driver.DMChainCanInterface(
        [(i, "DM4310") for i in range(1, 7)], np.zeros(6), np.ones(6), channel="yam_left", start_thread=False
    )
    filters = interface.bus.set_filters.call_args.args[0]
    assert {item["can_id"] for item in filters} == set(range(0x11, 0x17))
    assert all(item["can_mask"] == 0x7FF and not item["extended"] for item in filters)


def test_big_yam_factory_wraps_reversed_joint_and_retains_vendor_gravity(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = np.array([0.0, -6.4, 0.0, 0.0, 0.0, 0.0])
    direction = np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0])
    chain = Mock(motor_direction=direction, motor_offset=np.zeros(6))
    chain.read_states.side_effect = lambda: [SimpleNamespace(pos=p) for p in (raw - chain.motor_offset) * direction]
    factory = Mock(return_value=chain)
    robot = Mock()
    monkeypatch.setattr(get_robot, "DMChainCanInterface", factory)
    monkeypatch.setattr(get_robot, "MotorChainRobot", robot)
    get_robot.get_yam_robot(arm_type=ArmType.BIG_YAM, external_gripper=True, repair_motor_config=False)
    assert chain.read_states()[1].pos == pytest.approx(6.4 - 2 * np.pi)
    factory.assert_called_once()
    chain.start_thread.assert_called_once()
    assert factory.call_args.kwargs["repair_motor_config"] is False
    assert factory.call_args.kwargs["enable_auto_recovery"] is False
    np.testing.assert_array_equal(robot.call_args.kwargs["gravity_comp_factor"], [1.0, 1.2, 1.0, 1.1, 1.0, 1.0])
    np.testing.assert_array_equal(robot.call_args.kwargs["kp"], [80.0, 80.0, 80.0, 40.0, 40.0, 10.0])
    np.testing.assert_array_equal(robot.call_args.kwargs["kd"], [5.0, 5.0, 5.0, 3.0, 1.5, 1.5])


@pytest.mark.parametrize("stage", ["read_states", "start_thread", "robot"])
def test_factory_closes_chain_if_post_enable_startup_fails(monkeypatch: pytest.MonkeyPatch, stage: str) -> None:
    chain = Mock(motor_direction=np.ones(6), motor_offset=np.zeros(6))
    chain.read_states.return_value = [SimpleNamespace(pos=0.0) for _ in range(6)]
    robot = Mock()
    (robot if stage == "robot" else getattr(chain, stage)).side_effect = RuntimeError("startup failed")
    monkeypatch.setattr(get_robot, "DMChainCanInterface", Mock(return_value=chain))
    monkeypatch.setattr(get_robot, "MotorChainRobot", robot)
    with pytest.raises(RuntimeError, match="startup failed"):
        get_robot.get_yam_robot(arm_type=ArmType.BIG_YAM, external_gripper=True)
    chain.close.assert_called_once()


def test_partial_arm_startup_disables_every_motor_and_closes_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    interface = Mock()
    interface.motor_on.side_effect = [SimpleNamespace(position=0), RuntimeError("motor fault")]
    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", Mock(return_value=interface))
    with pytest.raises(RuntimeError, match="motor fault"):
        dm_driver.DMChainCanInterface(
            [(i, "DM4310") for i in range(1, 7)], np.zeros(6), np.ones(6), channel="yam_left", start_thread=False
        )
    assert [c.args[0] for c in interface.motor_off.call_args_list] == list(range(1, 7))
    interface.close.assert_called_once()


@pytest.mark.parametrize("fault", [False, True])
def test_arm_shutdown_disables_remaining_motors_even_if_one_fails(fault: int) -> None:
    chain = object.__new__(dm_driver.DMChainCanInterface)
    chain._closed, chain.running, chain._thread = False, True, None
    chain.motor_list = [(i, "DM4310") for i in range(1, 7)]
    chain.motor_interface = Mock()
    if fault:
        chain.motor_interface.motor_off.side_effect = [RuntimeError("no reply"), None, None, None, None, None]
        with pytest.raises(RuntimeError, match="disable failed"):
            chain.close()
    else:
        chain.close()
    assert not chain.running
    assert [c.args[0] for c in chain.motor_interface.motor_off.call_args_list] == list(range(1, 7))
    chain.motor_interface.close.assert_called_once()
    if fault:
        with pytest.raises(RuntimeError, match="disable failed"):
            chain.close()  # Preserve failures from control-thread cleanup.
    else:
        chain.close()


def test_enable_is_bounded_and_does_not_clear_unknown_faults() -> None:
    motor = object.__new__(dm_driver.DMSingleMotorCanInterface)
    motor._send_message_get_response = Mock()
    motor.parse_recv_message = Mock(return_value=SimpleNamespace(error_code="0x3", error_message="unknown"))
    motor.clean_error = Mock()
    with pytest.raises(RuntimeError, match="refused enable"):
        motor.motor_on(1, "DM4340")
    motor.clean_error.assert_not_called()
    motor.parse_recv_message.return_value = SimpleNamespace(error_code="0x0", error_message="disabled")
    motor._send_message_get_response.reset_mock()
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        motor.motor_on(1, "DM4340")
    assert motor._send_message_get_response.call_count == 3


def test_watchdog_shutdown_clears_latch_without_reenabling() -> None:
    motor = object.__new__(dm_driver.DMSingleMotorCanInterface)
    motor.cmd_idoffset = 0
    motor._send_message_get_response = Mock(
        side_effect=[SimpleNamespace(data=bytes([0xD1] * 8)), SimpleNamespace(data=bytes([0x01] * 8))]
    )
    motor.clean_error = Mock()
    motor.try_receive_message = Mock()
    motor.motor_off(1)
    motor.clean_error.assert_called_once_with(1)
    assert all(call.args[2] == [255] * 7 + [253] for call in motor._send_message_get_response.call_args_list)


def test_watchdog_clear_consumes_its_reply_before_enabling() -> None:
    """Three unpaired clear commands leave disabled replies ahead of enable."""
    from collections import deque

    import can

    from i2rt.motor_drivers.utils import ReceiveMode

    replies, commands = deque(), []
    state = 13  # communication watchdog

    def send(message: Any) -> Any:
        nonlocal state
        command = "status" if message.arbitration_id == 0x7FF else message.data[-1]
        commands.append(command)
        if command == 0xFB:
            state = 0
        elif command == 0xFC and state != 13:
            state = 1
        elif command == 0xFD and state != 13:
            state = 0
        replies.append(
            can.Message(arbitration_id=0x11, is_extended_id=False, data=[state << 4 | 1] + [0] * 5 + [31, 30])
        )

    motor = object.__new__(dm_driver.DMSingleMotorCanInterface)
    motor.bus = SimpleNamespace(
        send=send, recv=lambda **_: replies.popleft() if replies else None, channel_info="simulated yam_left"
    )
    motor.use_buffered_reader = False
    motor.receive_mode = ReceiveMode.p16
    motor.cmd_idoffset = 0
    motor.name = "simulated"
    assert motor.motor_on(1, "DM4340").error_code == "0x1"
    assert commands == [0xFC, "status", "status", 0xFD, 0xFB, 0xFD, "status", 0xFC]
    assert not replies
    motor.motor_off(1)
    assert state == 0 and not replies


def test_disable_retries_a_delayed_enabled_reply() -> None:
    motor = object.__new__(dm_driver.DMSingleMotorCanInterface)
    motor.cmd_idoffset = 0
    motor._send_message_get_response = Mock(
        side_effect=[SimpleNamespace(data=bytes([0x11] * 8)), SimpleNamespace(data=bytes([0x01] * 8))]
    )
    motor.clean_error = Mock()
    motor.motor_off(1)
    assert motor._send_message_get_response.call_count == 2
    motor.clean_error.assert_not_called()


@pytest.mark.parametrize("fault", [0x2, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE])
@pytest.mark.parametrize("clear_to,enable_to", [(0, 1), (0xC, 1), (0xD, 1), (0, 0xC)])
def test_startup_fault_reset_consumes_replies_and_recovers_only_once(
    fault: int, clear_to: int, enable_to: int
) -> None:
    from collections import deque

    import can

    from i2rt.motor_drivers.utils import ReceiveMode

    replies, commands = deque(), []
    state = fault

    def send(message: Any) -> Any:
        nonlocal state
        command = "status" if message.arbitration_id == 0x7FF else message.data[-1]
        commands.append(command)
        if command == "status":
            assert list(message.data) == [2, 0, 0xCC, 0, 0, 0, 0, 0]
        elif command == 0xFB:
            state = clear_to
        elif command == 0xFC and state == 0:
            state = enable_to
        elif command == 0xFD and state == 1:
            state = 0
        replies.append(
            can.Message(
                arbitration_id=0x12, is_extended_id=False, data=[state << 4 | 2, 0x80, 0, 0x80, 0x08, 0, 31, 30]
            )
        )

    motor = object.__new__(dm_driver.DMSingleMotorCanInterface)
    motor.bus = SimpleNamespace(
        send=send, recv=lambda **_: replies.popleft() if replies else None, channel_info="simulated yam_right"
    )
    motor.use_buffered_reader = False
    motor.receive_mode = ReceiveMode.p16
    motor.cmd_idoffset = 0
    motor.name = "simulated"
    if clear_to == 0 and enable_to == 1:
        assert motor.motor_on(2, "DM4340").error_code == "0x1"
        assert commands == [0xFC, "status", "status", 0xFD, 0xFB, 0xFD, "status", 0xFC]
    else:
        with pytest.raises(RuntimeError, match=r"disable was not confirmed|refused enable"):
            motor.motor_on(2, "DM4340")
    assert commands.count(0xFB) == 1
    assert commands.count(0xFC) == (2 if clear_to == 0 else 1)
    assert not replies


@pytest.mark.parametrize(
    "code,mos,rotor",
    [
        ("0xc", 31, 51),
        ("0xc", 31, 101),
        ("0xb", 51, 30),
        ("0xc", 0, 30),
        ("0xc", 31, float("nan")),
        ("0xc", 31, float("inf")),
        ("0x8", 51, 30),
        ("0x9", 31, 51),
        ("0xa", 31, 51),
        ("0xe", 51, 30),
        ("0x2", 31, 51),
        ("0x3", 31, 30),
        ("0xf", 31, 30),
    ],
)
def test_startup_never_clears_hot_invalid_or_unrecognized_faults(code: str, mos: float, rotor: float) -> None:
    motor = object.__new__(dm_driver.DMSingleMotorCanInterface)
    info = SimpleNamespace(error_code=code, error_message="fault", temperature_mos=mos, temperature_rotor=rotor)
    motor._send_message_get_response = Mock()
    motor.parse_recv_message = Mock(return_value=info)
    motor.read_motor_status = Mock(return_value=info)
    motor.clean_error, motor.motor_off = Mock(), Mock()
    with pytest.raises(RuntimeError, match="refused enable"):
        motor.motor_on(2, "DM4340")
    motor.clean_error.assert_not_called()
    motor.motor_off.assert_not_called()
    assert motor._send_message_get_response.call_count == 1


def test_startup_recovery_checks_a_second_fresh_sample_before_clearing() -> None:
    motor = object.__new__(dm_driver.DMSingleMotorCanInterface)
    motor.read_motor_status = Mock(
        side_effect=[
            SimpleNamespace(error_code="0xc", temperature_mos=31, temperature_rotor=30),
            SimpleNamespace(error_code="0xc", temperature_mos=31, temperature_rotor=101),
        ]
    )
    motor.clean_error = Mock()
    assert not motor.recover_startup_fault(2, "DM4340")
    assert motor.read_motor_status.call_count == 2
    motor.clean_error.assert_not_called()


def test_dynamics_compile_before_any_motor_is_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    prepared = object()
    chain = Mock(motor_direction=np.ones(6), motor_offset=np.zeros(6))
    chain.read_states.return_value = [SimpleNamespace(pos=0.0) for _ in range(6)]

    def compile_model(path: str) -> object:
        events.append("compile")
        return prepared

    def open_chain(*args: object, **kwargs: object) -> Mock:
        assert events == ["compile"]
        events.append("enable")
        return chain

    robot = Mock()
    monkeypatch.setattr(get_robot, "MuJoCoKDL", compile_model)
    monkeypatch.setattr(get_robot, "DMChainCanInterface", open_chain)
    monkeypatch.setattr(get_robot, "MotorChainRobot", robot)
    get_robot.get_yam_robot(arm_type=ArmType.BIG_YAM, external_gripper=True)
    assert robot.call_args.kwargs["kinematics"] is prepared


def test_failed_dynamics_compile_does_not_enable_motors(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = Mock()
    monkeypatch.setattr(get_robot, "DMChainCanInterface", factory)
    monkeypatch.setattr(get_robot, "MuJoCoKDL", Mock(side_effect=RuntimeError("invalid dynamics")))
    with pytest.raises(RuntimeError, match="invalid dynamics"):
        get_robot.get_yam_robot(arm_type=ArmType.BIG_YAM, external_gripper=True)
    factory.assert_not_called()


def test_robot_reuses_precompiled_dynamics_without_stalling_can(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock

    from i2rt.robots import motor_chain_robot

    prepared = SimpleNamespace(joint_limits=np.tile([-3.0, 3.0], (6, 1)))
    chain = MagicMock()
    chain.__len__.return_value = 6
    monkeypatch.setattr(motor_chain_robot, "MuJoCoKDL", Mock(side_effect=AssertionError("compiled after enable")))
    monkeypatch.setattr(motor_chain_robot.threading, "Thread", Mock())
    monkeypatch.setattr(
        motor_chain_robot.MotorChainRobot,
        "_motor_state_to_joint_state",
        lambda *args: SimpleNamespace(pos=np.zeros(6)),
    )
    robot = motor_chain_robot.MotorChainRobot(chain, xml_path="precompiled.xml", kinematics=prepared)
    assert robot.kdl is prepared

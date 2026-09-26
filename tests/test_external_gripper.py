"""External grippers retain payload dynamics without sharing motor ownership."""
from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest

from i2rt.robots import get_robot
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
from i2rt.motor_drivers import dm_driver


def test_external_gripper_preserves_payload_and_owns_only_arm_motors(monkeypatch):
    chain = Mock()
    chain.read_states.return_value = [SimpleNamespace(pos=0.) for _ in range(6)]
    factory = Mock(return_value=chain)
    robot = Mock()
    monkeypatch.setattr(get_robot, 'DMChainCanInterface', factory)
    monkeypatch.setattr(get_robot, 'MotorChainRobot', robot)
    monkeypatch.setattr(get_robot.time, 'sleep', lambda _: None)
    get_robot.get_yam_robot(channel='yam_left', external_gripper=True)
    for call in factory.call_args_list:
        assert [entry[0] for entry in call.args[0]] == [1, 2, 3, 4, 5, 6]
    args = robot.call_args.kwargs
    assert 'gripper_index' not in args
    assert len(args['kp']) == len(args['kd']) == 6
    loaded = mujoco.MjModel.from_xml_path(args['xml_path'])
    bare = mujoco.MjModel.from_xml_path(combine_arm_and_gripper_xml(
        ArmType.YAM.get_xml_path(), GripperType.NO_GRIPPER.get_xml_path()))
    assert loaded.body_mass.sum() > bare.body_mass.sum()
    data = mujoco.MjData(loaded)
    mujoco.mj_inverse(loaded, data)
    assert np.all(np.isfinite(data.qfrc_inverse[:6]))


def test_arm_socket_filters_out_external_gripper_feedback(monkeypatch):
    interface = Mock()
    monkeypatch.setattr(dm_driver, 'DMSingleMotorCanInterface', Mock(return_value=interface))
    monkeypatch.setattr(dm_driver.DMChainCanInterface, '_motor_on', lambda self: setattr(self, 'state', []))
    dm_driver.DMChainCanInterface([(i, 'DM4310') for i in range(1, 7)],
                                 np.zeros(6), np.ones(6), channel='yam_left', start_thread=False)
    filters = interface.bus.set_filters.call_args.args[0]
    assert {item['can_id'] for item in filters} == set(range(0x11, 0x17))
    assert all(item['can_mask'] == 0x7FF and not item['extended'] for item in filters)


def test_partial_arm_startup_disables_every_motor_and_closes_socket(monkeypatch):
    interface = Mock()
    interface.motor_on.side_effect = [SimpleNamespace(position=0), RuntimeError('motor fault')]
    monkeypatch.setattr(dm_driver, 'DMSingleMotorCanInterface', Mock(return_value=interface))
    with pytest.raises(RuntimeError, match='motor fault'):
        dm_driver.DMChainCanInterface([(i, 'DM4310') for i in range(1, 7)],
                                     np.zeros(6), np.ones(6), channel='yam_left', start_thread=False)
    assert [c.args[0] for c in interface.motor_off.call_args_list] == list(range(1, 7))
    interface.close.assert_called_once()


@pytest.mark.parametrize('fault', [False, True])
def test_arm_shutdown_disables_remaining_motors_even_if_one_fails(fault):
    chain = object.__new__(dm_driver.DMChainCanInterface)
    chain._closed, chain.running, chain._thread = False, True, None
    chain.motor_list = [(i, 'DM4310') for i in range(1, 7)]
    chain.motor_interface = Mock()
    if fault:
        chain.motor_interface.motor_off.side_effect = [RuntimeError('no reply'), None, None, None, None, None]
        with pytest.raises(RuntimeError, match='disable failed'):
            chain.close()
    else:
        chain.close()
    assert not chain.running
    assert [c.args[0] for c in chain.motor_interface.motor_off.call_args_list] == list(range(1, 7))
    chain.motor_interface.close.assert_called_once()
    if fault:
        with pytest.raises(RuntimeError, match='disable failed'):
            chain.close()  # Preserve failures from control-thread cleanup.
    else:
        chain.close()


def test_enable_is_bounded_and_does_not_clear_unknown_faults():
    motor = object.__new__(dm_driver.DMSingleMotorCanInterface)
    motor._send_message_get_response = Mock()
    motor.parse_recv_message = Mock(return_value=SimpleNamespace(error_code='0x3', error_message='unknown'))
    motor.clean_error = Mock()
    with pytest.raises(RuntimeError, match='refused enable'):
        motor.motor_on(1, 'DM4340')
    motor.clean_error.assert_not_called()
    motor.parse_recv_message.return_value = SimpleNamespace(error_code='0x0', error_message='disabled')
    motor._send_message_get_response.reset_mock()
    with pytest.raises(RuntimeError, match='after 3 attempts'):
        motor.motor_on(1, 'DM4340')
    assert motor._send_message_get_response.call_count == 3


def test_watchdog_shutdown_clears_latch_without_reenabling():
    motor = object.__new__(dm_driver.DMSingleMotorCanInterface)
    motor.cmd_idoffset = 0
    motor._send_message_get_response = Mock(side_effect=[SimpleNamespace(data=bytes([0xD1]*8)),
                                                        SimpleNamespace(data=bytes([0x01]*8))])
    motor.clean_error = Mock()
    motor.try_receive_message = Mock()
    motor.motor_off(1)
    motor.clean_error.assert_called_once_with(1)
    assert all(call.args[2] == [255]*7+[253] for call in motor._send_message_get_response.call_args_list)


def test_watchdog_clear_consumes_its_reply_before_enabling():
    """Three unpaired clear commands leave disabled replies ahead of enable."""
    from collections import deque
    import can
    from i2rt.motor_drivers.utils import ReceiveMode

    replies, commands = deque(), []
    state = 13  # communication watchdog

    def send(message):
        nonlocal state
        command = 'status' if message.arbitration_id == 0x7FF else message.data[-1]
        commands.append(command)
        if command == 0xFB:
            state = 0
        elif command == 0xFC and state != 13:
            state = 1
        elif command == 0xFD and state != 13:
            state = 0
        replies.append(can.Message(arbitration_id=0x11, data=[state << 4 | 1] + [0]*5 + [31, 30]))

    motor = object.__new__(dm_driver.DMSingleMotorCanInterface)
    motor.bus = SimpleNamespace(send=send, recv=lambda **_: replies.popleft() if replies else None,
                                channel_info='simulated yam_left')
    motor.use_buffered_reader = False
    motor.receive_mode = ReceiveMode.p16
    motor.cmd_idoffset = 0
    motor.name = 'simulated'
    assert motor.motor_on(1, 'DM4340').error_code == '0x1'
    assert commands == [0xFC, 'status', 'status', 0xFD, 0xFB, 0xFD, 'status', 0xFC]
    assert not replies
    motor.motor_off(1)
    assert state == 0 and not replies


def test_disable_retries_a_delayed_enabled_reply():
    motor = object.__new__(dm_driver.DMSingleMotorCanInterface)
    motor.cmd_idoffset = 0
    motor._send_message_get_response = Mock(side_effect=[SimpleNamespace(data=bytes([0x11]*8)),
                                                        SimpleNamespace(data=bytes([0x01]*8))])
    motor.clean_error = Mock()
    motor.motor_off(1)
    assert motor._send_message_get_response.call_count == 2
    motor.clean_error.assert_not_called()


@pytest.mark.parametrize('fault', [0x2, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE])
@pytest.mark.parametrize('clear_to,enable_to', [(0, 1), (0xC, 1), (0xD, 1), (0, 0xC)])
def test_startup_fault_reset_consumes_replies_and_recovers_only_once(fault, clear_to, enable_to):
    from collections import deque
    import can
    from i2rt.motor_drivers.utils import ReceiveMode

    replies, commands = deque(), []
    state = fault

    def send(message):
        nonlocal state
        command = 'status' if message.arbitration_id == 0x7FF else message.data[-1]
        commands.append(command)
        if command == 'status':
            assert list(message.data) == [2, 0, 0xCC, 0, 0, 0, 0, 0]
        elif command == 0xFB:
            state = clear_to
        elif command == 0xFC and state == 0:
            state = enable_to
        elif command == 0xFD and state == 1:
            state = 0
        replies.append(can.Message(arbitration_id=0x12,
                                   data=[state << 4 | 2, 0x80, 0, 0x80, 0x08, 0, 31, 30]))

    motor = object.__new__(dm_driver.DMSingleMotorCanInterface)
    motor.bus = SimpleNamespace(send=send, recv=lambda **_: replies.popleft() if replies else None,
                                channel_info='simulated yam_right')
    motor.use_buffered_reader = False
    motor.receive_mode = ReceiveMode.p16
    motor.cmd_idoffset = 0
    motor.name = 'simulated'
    if clear_to == 0 and enable_to == 1:
        assert motor.motor_on(2, 'DM4340').error_code == '0x1'
        assert commands == [0xFC, 'status', 'status', 0xFD, 0xFB, 0xFD, 'status', 0xFC]
    else:
        with pytest.raises(RuntimeError, match='disable was not confirmed|refused enable'):
            motor.motor_on(2, 'DM4340')
    assert commands.count(0xFB) == 1
    assert commands.count(0xFC) == (2 if clear_to == 0 else 1)
    assert not replies


@pytest.mark.parametrize('code,mos,rotor', [
    ('0xc', 31, 51), ('0xc', 31, 101), ('0xb', 51, 30),
    ('0xc', 0, 30), ('0xc', 31, float('nan')), ('0xc', 31, float('inf')),
    ('0x8', 51, 30), ('0x9', 31, 51), ('0xa', 31, 51), ('0xe', 51, 30), ('0x2', 31, 51),
    ('0x3', 31, 30), ('0xf', 31, 30),
])
def test_startup_never_clears_hot_invalid_or_unrecognized_faults(code, mos, rotor):
    motor = object.__new__(dm_driver.DMSingleMotorCanInterface)
    info = SimpleNamespace(error_code=code, error_message='fault', temperature_mos=mos, temperature_rotor=rotor)
    motor._send_message_get_response = Mock()
    motor.parse_recv_message = Mock(return_value=info)
    motor.read_motor_status = Mock(return_value=info)
    motor.clean_error, motor.motor_off = Mock(), Mock()
    with pytest.raises(RuntimeError, match='refused enable'):
        motor.motor_on(2, 'DM4340')
    motor.clean_error.assert_not_called()
    motor.motor_off.assert_not_called()
    assert motor._send_message_get_response.call_count == 1


def test_startup_recovery_checks_a_second_fresh_sample_before_clearing():
    motor = object.__new__(dm_driver.DMSingleMotorCanInterface)
    motor.read_motor_status = Mock(side_effect=[
        SimpleNamespace(error_code='0xc', temperature_mos=31, temperature_rotor=30),
        SimpleNamespace(error_code='0xc', temperature_mos=31, temperature_rotor=101),
    ])
    motor.clean_error = Mock()
    assert not motor.recover_startup_fault(2, 'DM4340')
    assert motor.read_motor_status.call_count == 2
    motor.clean_error.assert_not_called()

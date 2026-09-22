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
    motor.parse_recv_message = Mock(return_value=SimpleNamespace(error_code='0x2', error_message='unknown'))
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

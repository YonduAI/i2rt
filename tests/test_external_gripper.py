"""External grippers retain payload dynamics without sharing motor ownership."""
from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np

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

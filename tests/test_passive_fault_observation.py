"""Supported-bench observation must never send cleanup/enable/recovery frames."""
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import can
import numpy as np
import pytest

from i2rt.motor_drivers.can_interface import CanInterface
from i2rt.motor_drivers.dm_driver import DMChainCanInterface, MotorCmd
from i2rt.motor_drivers.utils import ReceiveMode


def chain_for_test(observe: bool) -> DMChainCanInterface:
    chain = object.__new__(DMChainCanInterface)
    chain.channel = 'FAKE'
    chain.running = True
    chain.observe_faults = observe
    chain.passive_fault = None
    chain.passive_feedback = {}
    chain._observation_stop = threading.Event()
    chain.command_lock = threading.Lock()
    chain.motor_list = [(1, 'DM4310')]
    chain._closed, chain._thread = False, None
    chain.commands = [MotorCmd(pos=1.0)]
    chain.motor_interface = Mock(receive_mode=ReceiveMode.p16)
    chain._set_torques_and_update_state = Mock(side_effect=TimeoutError('missing reply'))
    return chain


def test_default_fault_still_disables_and_closes() -> None:
    chain = chain_for_test(False)
    with pytest.raises(TimeoutError):
        chain._control_loop()
    chain.motor_interface.motor_off.assert_called_once_with(1)
    chain.motor_interface.close.assert_called_once()


def test_passive_fault_keeps_receiving_but_never_transmits_or_resumes() -> None:
    chain = chain_for_test(True)
    received = threading.Event()
    count = [0]

    def receive(**kwargs: object) -> can.Message | None:
        count[0] += 1
        if count[0] == 1:
            return can.Message(arbitration_id=0x11, data=bytes([0x11, 0, 0, 0, 0, 0, 25, 25]),
                               is_extended_id=False, timestamp=time.time())
        received.set()
        chain._observation_stop.wait(.01)
        return None

    chain.motor_interface._receive_message.side_effect = receive
    chain.motor_interface.parse_recv_message.return_value = SimpleNamespace(
        id=1, error_code='0x1', position=.3, temperature_mos=25., temperature_rotor=25.)
    chain._thread = threading.Thread(target=chain._control_loop)
    chain._thread.start()
    try:
        assert received.wait(1.)
        assert chain.passive_fault == 'missing reply' and chain.running is False
        assert chain.passive_feedback['1']['status'] == '0x1'
        chain.set_commands(np.zeros(1), pos=np.array([2.0]), get_state=False)
        assert chain.commands[0].pos == 1.0
        chain.motor_interface.motor_off.assert_not_called()
        chain.motor_interface.close.assert_not_called()
    finally:
        chain.close()
    assert not chain._thread.is_alive()
    chain.motor_interface.motor_off.assert_not_called()
    chain.motor_interface.close.assert_called_once()


def test_passive_gate_rejects_pending_retry_before_send() -> None:
    interface = object.__new__(CanInterface)
    interface.receive_mode = ReceiveMode.p16
    interface.name = 'FAKE'
    interface.bus = Mock()
    interface.transmit_suspended = True
    with pytest.raises(RuntimeError, match='suspended'):
        interface._send_message_get_response(1, 1, [0] * 8)
    interface.bus.send.assert_not_called()


def test_passive_latch_requires_explicit_opt_in() -> None:
    chain = chain_for_test(False)
    with pytest.raises(RuntimeError, match='not armed'):
        chain.enter_passive_fault('gripper timeout')
    assert chain.running and chain.passive_fault is None


def test_control_exception_publishes_passive_fault_before_cleanup_can_observe_stop() -> None:
    chain = chain_for_test(True)
    chain._rate_recorder = nullcontext()
    chain._report_interval = 30.
    chain._set_commands = Mock(side_effect=TimeoutError('missing motor 5'))
    with pytest.raises(TimeoutError):
        DMChainCanInterface._set_torques_and_update_state(chain)
    assert chain.running is False and chain.passive_fault == 'missing motor 5'
    assert chain.motor_interface.transmit_suspended is True
    chain.motor_interface.motor_off.assert_not_called()


def test_peer_fault_cancels_retry_after_first_send() -> None:
    interface = object.__new__(CanInterface)
    interface.receive_mode = ReceiveMode.p16
    interface.name = 'FAKE'
    interface.bus = Mock()
    interface.transmit_suspended = False

    def peer_fault(*args: object, **kwargs: object) -> None:
        interface.transmit_suspended = True

    interface._receive_message = peer_fault
    with pytest.raises(RuntimeError, match='suspended'):
        interface._send_message_get_response(1, 1, [0] * 8, deadline=time.monotonic() + .04)
    interface.bus.send.assert_called_once()

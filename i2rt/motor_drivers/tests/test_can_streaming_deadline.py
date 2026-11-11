"""Recover one lost reply without extending a motion batch's original deadline."""

from types import SimpleNamespace
from typing import Any

import can
import pytest

from i2rt.motor_drivers.can_interface import CanInterface
from i2rt.motor_drivers.dm_driver import ControlMode, DMSingleMotorCanInterface
from i2rt.motor_drivers.utils import MotorType, ReceiveMode


class Clock:
    now = 10.0

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Bus:
    channel_info = "fake-no-hardware"

    def __init__(self, clock: Clock, replies: list[tuple[float, can.Message | None]]) -> None:
        self.clock = clock
        self.replies = replies
        self.sent: list[can.Message] = []
        self.waits: list[float] = []

    def send(self, message: can.Message) -> None:
        self.sent.append(message)

    def recv(self, timeout: float) -> can.Message | None:
        self.waits.append(timeout)
        delay, reply = self.replies.pop(0) if self.replies else (timeout, None)
        self.clock.advance(delay)
        return reply


def interface(monkeypatch: pytest.MonkeyPatch, replies: list[Any]) -> tuple[CanInterface, Clock, Bus]:
    clock = Clock()
    monkeypatch.setattr("time.monotonic", lambda: clock.now)
    monkeypatch.setattr("time.monotonic_ns", lambda: int(clock.now * 1e9))
    monkeypatch.setattr("time.time", lambda: clock.now + 1000)
    monkeypatch.setattr("time.time_ns", lambda: int((clock.now + 1000) * 1e9))
    monkeypatch.setattr("time.sleep", clock.advance)
    obj = object.__new__(CanInterface)
    obj.bus = Bus(clock, replies)
    obj.receive_mode = ReceiveMode.p16
    obj.name = obj.channel = "fake-no-hardware"
    obj.use_buffered_reader = False
    return obj, clock, obj.bus


def reply(motor: int = 1) -> can.Message:
    return can.Message(
        arbitration_id=ReceiveMode.p16.get_receive_id(motor),
        is_extended_id=False,
        timestamp=1010.002,
        data=[0x10 | motor, 0x7F, 0xFF, 0x7F, 0xF7, 0xFF, 30, 30],
    )


def test_receive_uses_one_blocking_wait_instead_of_one_ms_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    obj, clock, bus = interface(monkeypatch, [])
    assert obj._receive_message(timeout=0.007, supress_warning=True) is None
    assert bus.waits == [0.007]
    assert clock.now == pytest.approx(10.007)


@pytest.mark.parametrize("expired", [False, True])
def test_stream_deadline_bounds_retry_and_prevents_expired_transmission(
    monkeypatch: pytest.MonkeyPatch, expired: bool
) -> None:
    obj, clock, bus = interface(monkeypatch, [])
    with pytest.raises(TimeoutError, match="deadline"):
        obj._send_message_get_response(1, 1, [0] * 8, max_retry=15, deadline=10.0 if expired else 10.01)
    assert len(bus.sent) == (0 if expired else 2)
    assert clock.now <= 10.01 + 1e-10


def test_reply_delivered_after_deadline_is_not_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    obj, _, bus = interface(monkeypatch, [(0.02, reply())])
    with pytest.raises(TimeoutError, match="deadline"):
        obj._send_message_get_response(1, 1, [0] * 8, deadline=10.01)
    assert len(bus.sent) == 1


def test_unrelated_frames_do_not_resend_or_renew_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    obj, clock, bus = interface(monkeypatch, [(0.002, reply(7)), (0.002, reply())])
    assert obj._send_message_get_response(1, 1, [0] * 8, deadline=10.01).arbitration_id == 0x11
    assert len(bus.sent) == 1
    assert bus.waits == pytest.approx([0.005, 0.003])
    assert clock.now == pytest.approx(10.004)


def test_stream_encoding_passes_deadline_with_one_bounded_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    obj = object.__new__(DMSingleMotorCanInterface)
    obj.control_mode = ControlMode.MIT
    obj.cmd_idoffset = 0
    obj._send_message_get_response = lambda *args, **kw: calls.append(kw) or reply()
    obj.parse_recv_message = lambda *args: SimpleNamespace(error_code="0x1")
    obj.set_control(1, MotorType.DM4310, 0.0, 0.0, 0.0, 0.0, 0.0, deadline=10.02)
    assert calls == [{"max_retry": 2, "deadline": 10.02}]


def test_whole_chain_has_one_budget_and_never_advances_after_missing_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from i2rt.motor_drivers.dm_driver import DMChainCanInterface, MotorCmd

    obj, clock, bus = interface(monkeypatch, [(0.002, reply())])
    obj.__class__ = DMSingleMotorCanInterface
    obj.control_mode = ControlMode.MIT
    obj.cmd_idoffset = 0
    chain = object.__new__(DMChainCanInterface)
    chain.motor_interface = obj
    chain.motor_list = [(i, MotorType.DM4310) for i in range(1, 7)]
    chain.motor_offset = np.zeros(6)
    chain.motor_direction = np.ones(6)
    chain.channel = "fake"
    chain.control_cycle_timeout = 0.01
    chain.last_applied_command = None
    commands = [MotorCmd(pos=0.0, vel=0.0, kp=0.0, kd=0.0, torque=0.0) for _ in range(6)]
    with pytest.raises(TimeoutError, match="motor 2"):
        chain._set_commands(commands)
    assert [m.arbitration_id for m in bus.sent] == [1, 2, 2]
    assert clock.now == pytest.approx(10.01)
    assert chain.last_applied_command is None


def test_lost_reply_recovers_same_target_without_enabling_or_extending_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obj, clock, bus = interface(monkeypatch, [(0.005, None), (0.001, reply())])
    payload = [0x7F, 0xFF, 0x7F, 0xF0, 0, 0, 7, 0xFF]
    assert obj._send_message_get_response(1, 1, payload, deadline=10.04).arbitration_id == 0x11
    assert [bytes(m.data) for m in bus.sent] == [bytes(payload)] * 2
    assert clock.now == pytest.approx(10.006)
    assert obj.last_transaction["attempts"] == 2


@pytest.mark.parametrize("noise", ["stuff", "short", "stale"])
def test_malformed_or_stale_reply_is_ignored_when_valid_reply_follows(
    monkeypatch: pytest.MonkeyPatch, noise: str
) -> None:
    bad = reply()
    if noise == "stuff":
        # Exact first error notification captured on 1commerce at 22:45:10.010071.
        bad = can.Message(
            arbitration_id=0x8, is_error_frame=True, is_extended_id=False, data=bytes.fromhex("0000040000000001")
        )
    elif noise == "short":
        bad.data = bytearray([0x11])
        bad.dlc = 1
    else:
        bad.timestamp = 1009.9
    obj, clock, bus = interface(monkeypatch, [(0.001, bad), (0.001, reply())])
    received = obj._send_message_get_response(1, 1, [0] * 8, deadline=10.04)
    assert received is not bad
    assert clock.now == pytest.approx(10.002)
    assert len(bus.sent) == 1


def test_repeated_error_frames_do_not_renew_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    error = can.Message(arbitration_id=0x24, is_error_frame=True, data=bytes.fromhex("0020000000008000"))
    obj, clock, bus = interface(monkeypatch, [(0.001, error)] * 100)
    with pytest.raises(TimeoutError, match="deadline"):
        obj._send_message_get_response(1, 1, [0] * 8, deadline=10.04)
    assert clock.now <= 10.041 + 1e-10
    assert len(bus.sent) == 2


@pytest.mark.parametrize("status", [0x0, 0xB, 0xD])
def test_disabled_thermal_and_watchdog_states_are_not_retried_or_ignored(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    fault = reply()
    fault.data[0] = (status << 4) | 1
    obj, _, bus = interface(monkeypatch, [(0.001, fault)])
    obj.__class__ = DMSingleMotorCanInterface
    obj.control_mode = ControlMode.MIT
    obj.cmd_idoffset = 0
    with pytest.raises(RuntimeError, match="Motor error detected"):
        obj.set_control(1, MotorType.DM4310, 0.0, 0.0, 0.0, 0.0, 0.0, deadline=10.04)
    assert len(bus.sent) == 1


def test_can_cadence_matches_configured_frequency_without_catchup(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading
    from contextlib import nullcontext

    from i2rt.motor_drivers.dm_driver import DMChainCanInterface

    _, clock, _ = interface(monkeypatch, [])
    chain = object.__new__(DMChainCanInterface)
    chain.control_period = 0.004
    chain.command_lock = threading.Lock()
    chain.state_lock = threading.Lock()
    chain.commands = []
    chain.running = True
    chain.motor_list = [1]
    chain._report_interval = 30.0
    chain._rate_recorder = nullcontext()
    chain._rate_recorder.track = lambda: None
    chain.same_bus_device_driver = None
    chain._update_absolute_positions = lambda feedback: None
    started: list[float] = []

    def send(commands: Any) -> list[Any]:
        started.append(clock.now)
        clock.advance(0.020 if len(started) == 3 else 0.001)
        chain.running = len(started) < 5
        return [SimpleNamespace(error_code="0x1")]

    chain._set_commands = send
    chain._set_torques_and_update_state()
    assert started == pytest.approx([10.0, 10.004, 10.008, 10.028, 10.032])


def test_success_preserves_send_kernel_receive_and_resume_timestamps(monkeypatch: pytest.MonkeyPatch) -> None:
    obj, _, _ = interface(monkeypatch, [(0.003, reply())])
    obj._send_message_get_response(1, 1, [0] * 8, deadline=10.01)
    timing = obj.last_transaction
    assert timing["motor_id"] == 1
    assert timing["send_monotonic_ns"] == 10_000_000_000
    assert timing["reply_timestamp_ns"] == 1_010_002_000_000
    assert timing["completed_timestamp_ns"] - timing["reply_timestamp_ns"] == pytest.approx(1_000_000, abs=1)

"""CAN reply routing without physical hardware."""

from collections import deque
from typing import Any
from unittest.mock import Mock

import can
import pytest

from i2rt.motor_drivers.can_interface import CanInterface
from i2rt.motor_drivers.utils import ReceiveMode


@pytest.mark.parametrize("expected_id", [None, 0, 0x31])
@pytest.mark.parametrize("noise", ["gripper", "error", "remote", "extended"])
def test_other_motor_traffic_does_not_discard_expected_reply(expected_id: int | None, noise: str) -> None:
    interface = object.__new__(CanInterface)
    interface.name = "arm"
    interface.receive_mode = ReceiveMode.p16
    interface.use_buffered_reader = False
    interface.bus = Mock(channel_info="fake")
    frames = deque()
    reply = can.Message(
        arbitration_id=0x11 if expected_id is None else expected_id, data=[1] * 8, is_extended_id=False
    )

    def send(_: Any) -> Any:
        # A separate gripper owner can transmit just before this arm's reply.
        if noise == "gripper":
            unwanted = can.Message(arbitration_id=7, data=[0] * 8, is_extended_id=False)
        else:
            unwanted = can.Message(
                arbitration_id=reply.arbitration_id,
                data=[1] * 8,
                is_error_frame=noise == "error",
                is_remote_frame=noise == "remote",
                is_extended_id=noise == "extended",
            )
        frames.extend([unwanted, reply])

    interface.bus.send.side_effect = send
    interface.bus.recv.side_effect = lambda **_: frames.popleft() if frames else None
    assert interface._send_message_get_response(1, 1, [0] * 8, expected_id=expected_id) is reply
    interface.bus.send.assert_called_once()


def test_missing_expected_reply_still_times_out_and_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import i2rt.motor_drivers.can_interface as module

    interface = object.__new__(CanInterface)
    interface.name, interface.receive_mode = "arm", ReceiveMode.p16
    interface.bus = Mock(channel_info="fake")
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", lambda _: None)

    def unrelated(*args: Any, **kwargs: Any) -> Any:
        clock[0] += 0.05
        return can.Message(arbitration_id=7, data=[0] * 8, is_extended_id=False)

    interface._receive_message = unrelated
    interface.try_receive_message = Mock()
    with pytest.raises(AssertionError, match="fail to communicate"):
        interface._send_message_get_response(1, 1, [0] * 8, max_retry=2)
    assert interface.bus.send.call_count == 2

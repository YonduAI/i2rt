"""A waiting CAN reply cannot block publishing a newer hold command."""

import threading
from types import SimpleNamespace
from typing import Any

import numpy as np

from i2rt.motor_drivers.dm_driver import DMChainCanInterface, MotorCmd
from i2rt.utils.utils import RateRecorder


def test_publisher_replaces_goal_while_an_older_can_cycle_is_blocked() -> None:
    chain = object.__new__(DMChainCanInterface)
    chain.channel = "FAKE"
    chain.running = True
    chain.motor_list = [(1, "DM4310")]
    chain.commands = [MotorCmd(pos=0.1)]
    chain.command_lock = threading.Lock()
    chain.state_lock = threading.Lock()
    chain.same_bus_device_driver = None
    chain._report_interval = 30.0
    chain._rate_recorder = RateRecorder(name="test", report_interval=30.0)
    chain._update_absolute_positions = lambda _: None
    entered, release, published = threading.Event(), threading.Event(), threading.Event()
    snapshots = []

    def send(commands: list[MotorCmd]) -> Any:
        snapshots.append(commands)
        entered.set()
        assert release.wait(2.0)
        chain.running = False
        return [SimpleNamespace(error_code="0x1")]

    chain._set_commands = send
    worker = threading.Thread(target=chain._set_torques_and_update_state)
    writer = None
    updated = False
    try:
        worker.start()
        assert entered.wait(1.0)

        def publish() -> None:
            chain.set_commands(np.zeros(1), pos=np.array([0.2]), get_state=False)
            published.set()

        writer = threading.Thread(target=publish)
        writer.start()
        updated = published.wait(0.25)
    finally:
        release.set()
        worker.join(2.0)
        if writer is not None:
            writer.join(2.0)
    assert updated, "CAN transaction held the publisher lock while waiting for a reply"
    assert snapshots[0][0].pos == 0.1 and chain.commands[0].pos == 0.2

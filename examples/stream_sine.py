"""
Example: drive the running pi-robotics simulation from a Python controller.

The daemon exposes a loopback JSON control API. Point this at the
`control_port` from `<project>/.pi/robotics/session.json` and stream actuator
targets while the live 3D panel updates in the browser.

    /robotics start arm3                     # in pi
    python examples/stream_sine.py           # in another shell
"""

from __future__ import annotations

import json
import math
import pathlib
import time
import urllib.request

SESSION = pathlib.Path(".pi/robotics/session.json")


def post(port: int, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as res:
        return json.loads(res.read())


def main() -> None:
    session = json.loads(SESSION.read_text())
    port = session["control_port"]
    print(f"driving control API on :{port}")

    # Joint-space sine sweep on the three arm joints.
    start = time.time()
    while time.time() - start < 8.0:
        t = time.time() - start
        for i, phase in enumerate((0.0, 0.7, 1.4)):
            value = 0.6 * math.sin(2.0 * t + phase)
            post(port, "/command", {"cmd": "ctrl_named", "name": f"joint{i}", "value": value})
        time.sleep(0.02)
    print("done")


if __name__ == "__main__":
    main()

"""CORS allowlist for the loopback control server."""

import pytest

from pi_robotics.server import loopback_origin


@pytest.mark.parametrize(
    "origin",
    [
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "https://localhost",
        "http://[::1]:9000",
    ],
)
def test_loopback_origins_are_allowed(origin):
    assert loopback_origin(origin) == origin


@pytest.mark.parametrize(
    "origin",
    [
        None,
        "",
        "https://evil.example.com",
        "http://192.168.1.10:8080",
        "ftp://127.0.0.1",
        "not a url",
    ],
)
def test_non_loopback_origins_are_rejected(origin):
    assert loopback_origin(origin) is None

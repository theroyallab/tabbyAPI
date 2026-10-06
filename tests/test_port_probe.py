"""The startup port check binds the configured address rather than connecting to localhost."""

import socket

import pytest

from common.networking import port_bind_error


@pytest.fixture
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_free_port_is_bindable(free_port):
    assert port_bind_error("127.0.0.1", free_port) is None
    assert port_bind_error("0.0.0.0", free_port) is None


def test_listener_on_the_same_address_is_reported(free_port):
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", free_port))
        taken.listen()
        error = port_bind_error("127.0.0.1", free_port)
        assert error is not None
        assert str(free_port) in error


def test_wildcard_listener_blocks_a_specific_bind(free_port):
    with socket.socket() as taken:
        taken.bind(("0.0.0.0", free_port))
        taken.listen()
        assert port_bind_error("127.0.0.1", free_port) is not None


def test_listener_on_another_interface_does_not_block():
    """
    The old probe connected to localhost, so a loopback-only listener (a Docker
    port mapping, a dev proxy) made the server move even when its own address
    was free. A bind on a different interface is unaffected by it.
    """

    # The address of whichever interface routes outward; nothing is sent
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))
            other = probe.getsockname()[0]
    except OSError:
        other = None
    if not other or other.startswith("127."):
        pytest.skip("no non-loopback IPv4 address on this machine")

    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        try:
            assert port_bind_error(other, port) is None
        except OSError as ex:  # pragma: no cover - sandboxed network
            pytest.skip(f"cannot bind {other}: {ex}")


def test_unresolvable_host_is_reported():
    error = port_bind_error("no.such.host.invalid", 5000)
    assert error is not None and "not a usable address" in error


def test_port_from_a_closed_listener_is_reusable(free_port):
    """A port left in TIME_WAIT must not count as taken (the server sets SO_REUSEADDR too)."""

    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", free_port))
        server.listen()
        with socket.create_connection(("127.0.0.1", free_port)) as client:
            conn, _ = server.accept()
            conn.close()
            client.close()
    assert port_bind_error("127.0.0.1", free_port) is None

from unittest import mock

from openpi_client import msgpack_numpy
from openpi_client import websocket_client_policy


def test_ping_interval_can_be_disabled() -> None:
    connection = mock.Mock()
    connection.recv.return_value = msgpack_numpy.Packer().pack({})

    with mock.patch(
        "openpi_client.websocket_client_policy.websockets.sync.client.connect",
        return_value=connection,
    ) as connect:
        websocket_client_policy.WebsocketClientPolicy(
            "127.0.0.1",
            8000,
            ping_interval=None,
        )

    connect.assert_called_once_with(
        "ws://127.0.0.1:8000",
        compression=None,
        max_size=None,
        additional_headers=None,
        ping_interval=None,
    )

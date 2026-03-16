"""Unit tests for tap_postgres.wal_prefetcher, covering WALPrefetcher's message parsing,
routing, buffering, and lifecycle logic using mocks (no running PostgreSQL instance required).
"""

import datetime
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tap_postgres.wal_prefetcher import (
    StreamWALConfig,
    WALMessage,
    WALPrefetcher,
    _escape_wal2json_table_fqn,
    _fix_wal2json_enum_quotes,
    _parse_wal_payload,
    _pre_parse_text_arrays,
)

# build mock objects that mimic psycopg2 replication messages/cursors


def _make_wal_message(payload_dict: dict, data_start: int = 100) -> SimpleNamespace:
    """Build a mock psycopg2 replication message.

    psycopg2's ReplicationMessage has .payload (str) and .data_start (int).
    """
    return SimpleNamespace(payload=json.dumps(payload_dict), data_start=data_start)


def _make_insert_payload(schema: str, table: str, columns: list[dict]) -> dict:
    """Build a wal2json format-version-2 insert payload."""
    return {"action": "I", "schema": schema, "table": table, "columns": columns}


def _make_delete_payload(schema: str, table: str, identity: list[dict]) -> dict:
    """Build a wal2json format-version-2 delete payload."""
    return {"action": "D", "schema": schema, "table": table, "identity": identity}


def _simple_columns(*values: tuple[str, str, object]) -> list[dict]:
    """Shorthand for building wal2json column lists; each value is (name, type, value)."""
    return [{"name": name, "type": typ, "value": val} for name, typ, val in values]


def _make_mock_cursor(messages: list[SimpleNamespace | None]):
    """Build a mock ReplicationCursor that yields the given message sequence.

    Messages are returned one-at-a-time from read_message(). A None entry
    causes read_message() to return None (simulating no-data-available),
    which the prefetcher handles via select.select(). After all entries
    are exhausted, read_message() returns None indefinitely.

    The mock also provides a no-op send_feedback and start_replication,
    and a feedback_timestamp that's always "now".
    """
    cursor = MagicMock()
    msg_iter = iter(messages)

    def _read_message():
        return next(msg_iter, None)

    cursor.read_message = _read_message
    cursor.feedback_timestamp = datetime.datetime.now()
    cursor.send_feedback = MagicMock()
    cursor.start_replication = MagicMock()
    cursor.close = MagicMock()
    # Make cursor work with select.select() — fileno is required
    cursor.fileno = MagicMock(return_value=0)
    return cursor


def _make_mock_connection(cursor):
    """Build a mock psycopg2 LogicalReplicationConnection."""
    conn = MagicMock()
    conn.cursor.return_value = cursor
    conn.close = MagicMock()
    return conn


class TestFixWal2jsonEnumQuotes:
    """Tests for _fix_wal2json_enum_quotes."""

    def test_fixes_double_quoted_enum_type(self):
        broken = '{"action":"I","columns":[{"name":"status","type":""MyEnum"","value":"active"}]}'
        fixed = _fix_wal2json_enum_quotes(broken)
        parsed = json.loads(fixed)
        assert parsed["columns"][0]["type"] == "MyEnum"

    def test_leaves_normal_types_unchanged(self):
        normal = '{"action":"I","columns":[{"name":"id","type":"integer","value":1}]}'
        assert _fix_wal2json_enum_quotes(normal) == normal

    def test_fixes_multiple_enum_columns(self):
        broken = (
            '{"columns":['
            '{"name":"a","type":""Enum1"","value":"x"},'
            '{"name":"b","type":""Enum2"","value":"y"}'
            "]}"
        )
        fixed = _fix_wal2json_enum_quotes(broken)
        parsed = json.loads(fixed)
        assert parsed["columns"][0]["type"] == "Enum1"
        assert parsed["columns"][1]["type"] == "Enum2"


@pytest.mark.parametrize(
    ["fqn", "exp_result"],
    [
        ("public.users", "public.users"),
        ("my schema.my table", r"my\ schema.my\ table"),
        ("a b'c,d.e*f", r"a\ b\'c\,d.e\*f"),
    ],
)
def test_escape_wal2json_table_name(fqn, exp_result):
    result = _escape_wal2json_table_fqn(fqn)
    assert result == exp_result


class TestPreParseTextArrays:
    """Tests for _pre_parse_text_arrays."""

    def test_converts_text_array_with_cursor(self):
        payload = {
            "columns": [
                {"name": "tags", "type": "text[]", "value": "{a,b,c}"},
                {"name": "id", "type": "integer", "value": 1},
            ]
        }
        mock_cursor = MagicMock()
        with patch("tap_postgres.wal_prefetcher.STRINGARRAY", return_value=["a", "b", "c"]):
            _pre_parse_text_arrays(payload, mock_cursor)

        assert payload["columns"][0]["value"] == ["a", "b", "c"]
        # Non-array column untouched
        assert payload["columns"][1]["value"] == 1

    def test_skips_null_text_array(self):
        payload = {
            "columns": [
                {"name": "tags", "type": "text[]", "value": None},
            ]
        }
        _pre_parse_text_arrays(payload, MagicMock())
        assert payload["columns"][0]["value"] is None

    def test_handles_identity_key_for_deletes(self):
        payload = {
            "identity": [
                {"name": "tags", "type": "text[]", "value": "{x}"},
            ]
        }
        with patch("tap_postgres.wal_prefetcher.STRINGARRAY", return_value=["x"]):
            _pre_parse_text_arrays(payload, MagicMock())

        assert payload["identity"][0]["value"] == ["x"]


class TestParseWalPayload:
    """Tests for _parse_wal_payload."""

    def test_parses_valid_json(self):
        raw = json.dumps({"action": "I", "schema": "public", "table": "t"})
        result = _parse_wal_payload(raw, cursor=None)
        assert result is not None
        assert result["action"] == "I"

    def test_returns_none_for_unparseable_json(self):
        result = _parse_wal_payload("not json at all {{{{", cursor=None)
        assert result is None

    def test_fixes_and_parses_enum_quoted_json(self):
        broken = '{"action":"I","schema":"public","table":"t","columns":[{"name":"s","type":""MyEnum"","value":"a"}]}'
        result = _parse_wal_payload(broken, cursor=None)
        assert result is not None
        assert result["columns"][0]["type"] == "MyEnum"


# ---------------------------------------------------------------------------
# WALMessage dataclass tests
# ---------------------------------------------------------------------------


class TestWALMessage:
    """Tests for the WALMessage dataclass."""

    def test_is_immutable(self):
        msg = WALMessage(table_fqn="public.users", action="I", lsn=100, payload={})
        with pytest.raises(AttributeError):
            msg.lsn = 200  # type: ignore[misc]

    def test_fields_accessible(self):
        payload = {"action": "I", "schema": "public", "table": "users"}
        msg = WALMessage(table_fqn="public.users", action="I", lsn=42, payload=payload)
        assert msg.table_fqn == "public.users"
        assert msg.action == "I"
        assert msg.lsn == 42
        assert msg.payload is payload


# ---------------------------------------------------------------------------
# WALPrefetcher tests
# ---------------------------------------------------------------------------


def _build_prefetcher(
    streams: dict[str, StreamWALConfig],
    messages: list[SimpleNamespace | None],
    status_interval: int = 5,
) -> WALPrefetcher:
    """Build a WALPrefetcher wired to a mock connection/cursor.

    Patches psycopg2.connect and select.select so that:
    - The cursor yields the given messages from read_message()
    - select.select() always returns empty (triggering the timeout exit)
      when the message sequence is exhausted
    """
    mock_cursor = _make_mock_cursor(messages)
    mock_conn = _make_mock_connection(mock_cursor)
    mock_conn_params = MagicMock()
    mock_conn_params.render_as_psycopg2_dsn.return_value = "host=localhost"

    prefetcher = WALPrefetcher(
        connection_parameters=mock_conn_params,
        streams=streams,
        replication_slot_name="test_slot",
        status_interval=status_interval,
    )

    return prefetcher, mock_conn, mock_cursor


class TestWALPrefetcherInit:
    """Tests for WALPrefetcher construction and validation."""

    def test_rejects_empty_streams(self):
        with pytest.raises(ValueError, match="at least one stream"):
            WALPrefetcher(
                connection_parameters=MagicMock(),
                streams={},
            )

    def test_creates_empty_buffers_for_each_stream(self):
        streams = {
            "public.a": StreamWALConfig("public.a", start_lsn=0),
            "public.b": StreamWALConfig("public.b", start_lsn=100),
        }
        pf = WALPrefetcher(connection_parameters=MagicMock(), streams=streams)
        assert set(pf._buffers.keys()) == {"public.a", "public.b"}
        assert all(buf == [] for buf in pf._buffers.values())


class TestWALPrefetcherRun:
    """Tests for WALPrefetcher.run() message routing and filtering."""

    def test_routes_messages_to_correct_stream_buffers(self):
        """Messages for different tables land in the right buffers."""
        streams = {
            "public.users": StreamWALConfig("public.users", start_lsn=0),
            "public.orders": StreamWALConfig("public.orders", start_lsn=0),
        }
        messages = [
            _make_wal_message(
                _make_insert_payload(
                    "public",
                    "users",
                    _simple_columns(("id", "integer", 1)),
                ),
                data_start=10,
            ),
            _make_wal_message(
                _make_insert_payload(
                    "public",
                    "orders",
                    _simple_columns(("id", "integer", 100)),
                ),
                data_start=20,
            ),
            _make_wal_message(
                _make_insert_payload(
                    "public",
                    "users",
                    _simple_columns(("id", "integer", 2)),
                ),
                data_start=30,
            ),
        ]

        prefetcher, mock_conn, mock_cursor = _build_prefetcher(streams, messages)

        with (
            patch(
                "tap_postgres.wal_prefetcher.psycopg2.connect",
                return_value=mock_conn,
            ),
            patch(
                "tap_postgres.wal_prefetcher.select_mod.select",
                return_value=([], [], []),
            ),
        ):
            prefetcher.run()

        user_msgs = prefetcher.get_messages("public.users")
        order_msgs = prefetcher.get_messages("public.orders")

        assert len(user_msgs) == 2
        assert len(order_msgs) == 1
        assert user_msgs[0].lsn == 10
        assert user_msgs[1].lsn == 30
        assert order_msgs[0].lsn == 20

    def test_filters_messages_below_stream_start_lsn(self):
        """Messages older than a stream's bookmark are discarded."""
        streams = {
            "public.users": StreamWALConfig("public.users", start_lsn=50),
        }
        messages = [
            _make_wal_message(
                _make_insert_payload(
                    "public",
                    "users",
                    _simple_columns(("id", "integer", 1)),
                ),
                data_start=10,  # < 50 → should be filtered
            ),
            _make_wal_message(
                _make_insert_payload(
                    "public",
                    "users",
                    _simple_columns(("id", "integer", 2)),
                ),
                data_start=50,  # == 50 → should be kept
            ),
            _make_wal_message(
                _make_insert_payload(
                    "public",
                    "users",
                    _simple_columns(("id", "integer", 3)),
                ),
                data_start=60,  # > 50 → should be kept
            ),
        ]

        prefetcher, mock_conn, _ = _build_prefetcher(streams, messages)

        with (
            patch(
                "tap_postgres.wal_prefetcher.psycopg2.connect",
                return_value=mock_conn,
            ),
            patch(
                "tap_postgres.wal_prefetcher.select_mod.select",
                return_value=([], [], []),
            ),
        ):
            prefetcher.run()

        user_msgs = prefetcher.get_messages("public.users")
        assert len(user_msgs) == 2
        assert user_msgs[0].lsn == 50
        assert user_msgs[1].lsn == 60

    def test_different_start_lsns_per_stream(self):
        """Each stream filters independently based on its own start_lsn."""
        streams = {
            "public.a": StreamWALConfig("public.a", start_lsn=10),
            "public.b": StreamWALConfig("public.b", start_lsn=30),
        }
        messages = [
            _make_wal_message(
                _make_insert_payload("public", "a", _simple_columns(("x", "integer", 1))),
                data_start=20,
            ),
            _make_wal_message(
                _make_insert_payload("public", "b", _simple_columns(("x", "integer", 2))),
                data_start=20,  # < 30 → filtered for b
            ),
            _make_wal_message(
                _make_insert_payload("public", "b", _simple_columns(("x", "integer", 3))),
                data_start=40,  # >= 30 → kept for b
            ),
        ]

        prefetcher, mock_conn, _ = _build_prefetcher(streams, messages)

        with (
            patch(
                "tap_postgres.wal_prefetcher.psycopg2.connect",
                return_value=mock_conn,
            ),
            patch(
                "tap_postgres.wal_prefetcher.select_mod.select",
                return_value=([], [], []),
            ),
        ):
            prefetcher.run()

        assert len(prefetcher.get_messages("public.a")) == 1
        assert len(prefetcher.get_messages("public.b")) == 1
        assert prefetcher.get_messages("public.b")[0].lsn == 40

    def test_discards_messages_for_unregistered_tables(self):
        """Messages for tables not in the streams dict are silently dropped."""
        streams = {
            "public.users": StreamWALConfig("public.users", start_lsn=0),
        }
        messages = [
            _make_wal_message(
                _make_insert_payload(
                    "public",
                    "unknown_table",
                    _simple_columns(("id", "integer", 1)),
                ),
                data_start=10,
            ),
            _make_wal_message(
                _make_insert_payload(
                    "public",
                    "users",
                    _simple_columns(("id", "integer", 1)),
                ),
                data_start=20,
            ),
        ]

        prefetcher, mock_conn, _ = _build_prefetcher(streams, messages)

        with (
            patch(
                "tap_postgres.wal_prefetcher.psycopg2.connect",
                return_value=mock_conn,
            ),
            patch(
                "tap_postgres.wal_prefetcher.select_mod.select",
                return_value=([], [], []),
            ),
        ):
            prefetcher.run()

        assert len(prefetcher.get_messages("public.users")) == 1

    def test_skips_transaction_and_truncate_actions(self):
        """B (begin), C (commit), T (truncate) messages are not buffered."""
        streams = {
            "public.users": StreamWALConfig("public.users", start_lsn=0),
        }
        messages = [
            _make_wal_message({"action": "B"}, data_start=1),
            _make_wal_message(
                _make_insert_payload(
                    "public",
                    "users",
                    _simple_columns(("id", "integer", 1)),
                ),
                data_start=10,
            ),
            _make_wal_message({"action": "C"}, data_start=11),
            _make_wal_message(
                {"action": "T", "schema": "public", "table": "users"},
                data_start=12,
            ),
        ]

        prefetcher, mock_conn, _ = _build_prefetcher(streams, messages)

        with (
            patch(
                "tap_postgres.wal_prefetcher.psycopg2.connect",
                return_value=mock_conn,
            ),
            patch(
                "tap_postgres.wal_prefetcher.select_mod.select",
                return_value=([], [], []),
            ),
        ):
            prefetcher.run()

        assert len(prefetcher.get_messages("public.users")) == 1

    def test_handles_delete_payloads(self):
        """Delete messages (action=D) use 'identity' instead of 'columns'."""
        streams = {
            "public.users": StreamWALConfig("public.users", start_lsn=0),
        }
        messages = [
            _make_wal_message(
                _make_delete_payload(
                    "public",
                    "users",
                    _simple_columns(("id", "integer", 42)),
                ),
                data_start=10,
            ),
        ]

        prefetcher, mock_conn, _ = _build_prefetcher(streams, messages)

        with (
            patch(
                "tap_postgres.wal_prefetcher.psycopg2.connect",
                return_value=mock_conn,
            ),
            patch(
                "tap_postgres.wal_prefetcher.select_mod.select",
                return_value=([], [], []),
            ),
        ):
            prefetcher.run()

        msgs = prefetcher.get_messages("public.users")
        assert len(msgs) == 1
        assert msgs[0].action == "D"
        assert msgs[0].payload["identity"][0]["value"] == 42

    def test_uses_global_min_lsn_for_start_replication(self):
        """start_replication should be called with min(start_lsn)."""
        streams = {
            "public.a": StreamWALConfig("public.a", start_lsn=100),
            "public.b": StreamWALConfig("public.b", start_lsn=50),
        }

        prefetcher, mock_conn, mock_cursor = _build_prefetcher(
            streams,
            [],  # no messages
        )

        with (
            patch(
                "tap_postgres.wal_prefetcher.psycopg2.connect",
                return_value=mock_conn,
            ),
            patch(
                "tap_postgres.wal_prefetcher.select_mod.select",
                return_value=([], [], []),
            ),
        ):
            prefetcher.run()

        # Verify start_replication was called with the minimum LSN
        call_kwargs = mock_cursor.start_replication.call_args
        assert call_kwargs.kwargs["start_lsn"] == 50

    def test_flushes_global_min_lsn_before_reading(self):
        """send_feedback should flush up to min(start_lsn) before reading."""
        streams = {
            "public.a": StreamWALConfig("public.a", start_lsn=200),
            "public.b": StreamWALConfig("public.b", start_lsn=100),
        }

        prefetcher, mock_conn, mock_cursor = _build_prefetcher(streams, [])

        with (
            patch(
                "tap_postgres.wal_prefetcher.psycopg2.connect",
                return_value=mock_conn,
            ),
            patch(
                "tap_postgres.wal_prefetcher.select_mod.select",
                return_value=([], [], []),
            ),
        ):
            prefetcher.run()

        mock_cursor.send_feedback.assert_called_once_with(flush_lsn=100)

    def test_add_tables_includes_all_streams(self):
        """start_replication should list all tables in add-tables."""
        streams = {
            "public.users": StreamWALConfig("public.users", start_lsn=0),
            "myschema.orders": StreamWALConfig("myschema.orders", start_lsn=0),
        }

        prefetcher, mock_conn, mock_cursor = _build_prefetcher(streams, [])

        with (
            patch(
                "tap_postgres.wal_prefetcher.psycopg2.connect",
                return_value=mock_conn,
            ),
            patch(
                "tap_postgres.wal_prefetcher.select_mod.select",
                return_value=([], [], []),
            ),
        ):
            prefetcher.run()

        call_kwargs = mock_cursor.start_replication.call_args.kwargs
        add_tables = call_kwargs["options"]["add-tables"]
        # Both tables should appear (order may vary)
        assert "public.users" in add_tables
        assert "myschema.orders" in add_tables

    def test_empty_wal_produces_empty_buffers(self):
        """If no WAL messages arrive, all buffers remain empty."""
        streams = {
            "public.users": StreamWALConfig("public.users", start_lsn=0),
        }

        prefetcher, mock_conn, _ = _build_prefetcher(streams, [])

        with (
            patch(
                "tap_postgres.wal_prefetcher.psycopg2.connect",
                return_value=mock_conn,
            ),
            patch(
                "tap_postgres.wal_prefetcher.select_mod.select",
                return_value=([], [], []),
            ),
        ):
            prefetcher.run()

        assert prefetcher.get_messages("public.users") == []

    def test_closes_cursor_and_connection(self):
        """Cursor and connection are closed even when no errors occur."""
        streams = {
            "public.users": StreamWALConfig("public.users", start_lsn=0),
        }

        prefetcher, mock_conn, mock_cursor = _build_prefetcher(streams, [])

        with (
            patch(
                "tap_postgres.wal_prefetcher.psycopg2.connect",
                return_value=mock_conn,
            ),
            patch(
                "tap_postgres.wal_prefetcher.select_mod.select",
                return_value=([], [], []),
            ),
        ):
            prefetcher.run()

        mock_cursor.close.assert_called_once()
        mock_conn.close.assert_called_once()

    def test_closes_cursor_and_connection_on_error(self):
        """Cleanup happens even when the read loop raises."""
        streams = {
            "public.users": StreamWALConfig("public.users", start_lsn=0),
        }

        prefetcher, mock_conn, mock_cursor = _build_prefetcher(streams, [])
        mock_cursor.start_replication.side_effect = RuntimeError("boom")

        with (
            patch(
                "tap_postgres.wal_prefetcher.psycopg2.connect",
                return_value=mock_conn,
            ),
            pytest.raises(RuntimeError, match="boom"),
        ):
            prefetcher.run()

        mock_cursor.close.assert_called_once()
        mock_conn.close.assert_called_once()

    def test_unparseable_message_is_skipped(self):
        """A message with invalid JSON is silently skipped."""
        streams = {
            "public.users": StreamWALConfig("public.users", start_lsn=0),
        }
        bad_msg = SimpleNamespace(
            payload="this is not json {{{",
            data_start=10,
        )
        good_msg = _make_wal_message(
            _make_insert_payload(
                "public",
                "users",
                _simple_columns(("id", "integer", 1)),
            ),
            data_start=20,
        )

        prefetcher, mock_conn, _ = _build_prefetcher(streams, [bad_msg, good_msg])

        with (
            patch(
                "tap_postgres.wal_prefetcher.psycopg2.connect",
                return_value=mock_conn,
            ),
            patch(
                "tap_postgres.wal_prefetcher.select_mod.select",
                return_value=([], [], []),
            ),
        ):
            prefetcher.run()

        assert len(prefetcher.get_messages("public.users")) == 1


class TestWALPrefetcherLifecycle:
    """Tests for WALPrefetcher lifecycle and error handling."""

    def test_run_cannot_be_called_twice(self):
        streams = {
            "public.users": StreamWALConfig("public.users", start_lsn=0),
        }
        prefetcher, mock_conn, _ = _build_prefetcher(streams, [])

        with (
            patch(
                "tap_postgres.wal_prefetcher.psycopg2.connect",
                return_value=mock_conn,
            ),
            patch(
                "tap_postgres.wal_prefetcher.select_mod.select",
                return_value=([], [], []),
            ),
        ):
            prefetcher.run()

        with pytest.raises(RuntimeError, match="must not be called twice"):
            prefetcher.run()

    def test_get_messages_before_run_raises(self):
        streams = {
            "public.users": StreamWALConfig("public.users", start_lsn=0),
        }
        pf = WALPrefetcher(connection_parameters=MagicMock(), streams=streams)

        with pytest.raises(RuntimeError, match="before run"):
            pf.get_messages("public.users")

    def test_get_messages_for_unregistered_table_raises(self):
        streams = {
            "public.users": StreamWALConfig("public.users", start_lsn=0),
        }
        prefetcher, mock_conn, _ = _build_prefetcher(streams, [])

        with (
            patch(
                "tap_postgres.wal_prefetcher.psycopg2.connect",
                return_value=mock_conn,
            ),
            patch(
                "tap_postgres.wal_prefetcher.select_mod.select",
                return_value=([], [], []),
            ),
        ):
            prefetcher.run()

        with pytest.raises(KeyError, match="not registered"):
            prefetcher.get_messages("public.nonexistent")

    def test_get_flush_lsn_returns_minimum(self):
        streams = {
            "public.a": StreamWALConfig("public.a", start_lsn=500),
            "public.b": StreamWALConfig("public.b", start_lsn=200),
            "public.c": StreamWALConfig("public.c", start_lsn=800),
        }
        pf = WALPrefetcher(connection_parameters=MagicMock(), streams=streams)
        assert pf.get_flush_lsn() == 200


class TestWALPrefetcherMessageOrdering:
    """Tests that verify messages are buffered in WAL (LSN) order."""

    def test_messages_preserved_in_lsn_order(self):
        """Messages should appear in buffers in the order they were read."""
        streams = {
            "public.users": StreamWALConfig("public.users", start_lsn=0),
        }
        messages = [
            _make_wal_message(
                _make_insert_payload(
                    "public",
                    "users",
                    _simple_columns(("id", "integer", i)),
                ),
                data_start=i * 10,
            )
            for i in range(1, 6)
        ]

        prefetcher, mock_conn, _ = _build_prefetcher(streams, messages)

        with (
            patch(
                "tap_postgres.wal_prefetcher.psycopg2.connect",
                return_value=mock_conn,
            ),
            patch(
                "tap_postgres.wal_prefetcher.select_mod.select",
                return_value=([], [], []),
            ),
        ):
            prefetcher.run()

        lsns = [m.lsn for m in prefetcher.get_messages("public.users")]
        assert lsns == [10, 20, 30, 40, 50]

    def test_interleaved_multi_table_preserves_per_stream_order(self):
        """When messages for multiple tables are interleaved in the WAL,
        each stream's buffer preserves its own LSN ordering."""
        streams = {
            "public.a": StreamWALConfig("public.a", start_lsn=0),
            "public.b": StreamWALConfig("public.b", start_lsn=0),
        }
        # Interleaved: a@10, b@20, a@30, b@40, a@50
        messages = [
            _make_wal_message(
                _make_insert_payload("public", "a", _simple_columns(("x", "integer", 1))),
                data_start=10,
            ),
            _make_wal_message(
                _make_insert_payload("public", "b", _simple_columns(("x", "integer", 2))),
                data_start=20,
            ),
            _make_wal_message(
                _make_insert_payload("public", "a", _simple_columns(("x", "integer", 3))),
                data_start=30,
            ),
            _make_wal_message(
                _make_insert_payload("public", "b", _simple_columns(("x", "integer", 4))),
                data_start=40,
            ),
            _make_wal_message(
                _make_insert_payload("public", "a", _simple_columns(("x", "integer", 5))),
                data_start=50,
            ),
        ]

        prefetcher, mock_conn, _ = _build_prefetcher(streams, messages)

        with (
            patch(
                "tap_postgres.wal_prefetcher.psycopg2.connect",
                return_value=mock_conn,
            ),
            patch(
                "tap_postgres.wal_prefetcher.select_mod.select",
                return_value=([], [], []),
            ),
        ):
            prefetcher.run()

        a_lsns = [m.lsn for m in prefetcher.get_messages("public.a")]
        b_lsns = [m.lsn for m in prefetcher.get_messages("public.b")]
        assert a_lsns == [10, 30, 50]
        assert b_lsns == [20, 40]

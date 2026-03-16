"""Single-pass WAL prefetcher for LOG_BASED replication.

Reads PostgreSQL WAL exactly once for all selected LOG_BASED streams,
partitions messages by table, and stores them in per-stream buffers
that ``PostgresLogBasedStream.get_records()`` can drain.
"""

import json
import re
import select as select_mod
from dataclasses import dataclass
from datetime import datetime, timezone

import psycopg2
from psycopg2 import extras
from psycopg2.extensions import STRINGARRAY

from .connection_parameters import ConnectionParameters

# wal2json emits enum type names with unescaped quotes: we need to strip them
# such that "type":""EnumName"" => "type":"EnumName"
_WAL2JSON_ENUM_QUOTE_RE = re.compile(r'"type":""([^"]+)""')

_WAL2JSON_CHARS_TO_ESCAPE: tuple[str, ...] = (" ", "'", ",", ".", "*")

_NUMERIC_DTYPES = {"int", "numeric", "decimal", "real", "double", "float", "bigint", "smallint"}
_UPSERT_ACTIONS = {"I", "U"}
_DELETE_ACTIONS = {"D"}
_SKIP_ACTIONS = {"B", "C", "T"}


@dataclass(frozen=True, slots=True)
class WALMessage:
    """A single, parsed WAL change, ready for stream consumption."""

    table_fqn: str  # "schema.table", used for dispatch
    action: str  # "I", "U", "D", "T", "B", "C"
    lsn: int  # msg.data_start
    payload: dict  # parsed wal2json dict


@dataclass(frozen=True)
class StreamWALConfig:
    """Per-stream configuration for the WAL prefetcher."""

    table_fqn: str  # fully qualified "schema.table"
    start_lsn: int  # stream's bookmark (0 if first sync)


class WALPrefetcher:
    """Read WAL once and partition messages by fully-qualified table name.

    Usage::
        prefetcher = WALPrefetcher(conn_params, streams, slot_name)
        prefetcher.run()

        # Later, from each stream's get_records():
        messages = prefetcher.get_messages("public.my_table")
    """

    def __init__(
        self,
        connection_parameters: ConnectionParameters,
        streams: dict[str, StreamWALConfig],
        replication_slot_name: str = "tappostgres",
        status_interval: int = 5,
    ) -> None:
        """Excessively strict ruff rules require this line.

        Args:
            connection_parameters: Shared connection details for ``psycopg2`` .
            streams: Mapping of fully_qualified_name => StreamWALConfig
                (start_lsn per stream, see below).
            replication_slot_name: Name of the PG replication slot.
            status_interval: Seconds to wait with no messages before stopping.
        """
        if not streams:
            raise ValueError("WALPrefetcher requires at least one stream")

        self._connection_parameters = connection_parameters
        self._streams = streams
        self._replication_slot_name = replication_slot_name
        self._status_interval = status_interval

        # per-stream message buffers, keyed by table_fqn
        # populated by run(), read by get_messages()
        self._buffers: dict[str, list[WALMessage]] = {fqn: [] for fqn in streams}

        # Set after run() completes.
        self._has_run = False

    def run(self) -> None:
        """Execute single-pass WAL read.

        Opens one ``LogicalReplicationConnection``, calls ``start_replication()``
        with "add-tables" listing *all* registered streams, reads messages until
        ``status_interval`` timeout, and partitions each message into the
        appropriate stream's buffer.

        Messages with LSN less than a stream's ``start_lsn`` are silently discarded
        for that stream (but may be relevant to other streams).

        Must be called exactly once before :meth:`get_messages()`.
        """
        if self._has_run:
            raise RuntimeError("WALPrefetcher.run() must not be called twice")

        # "global" start_lsn is the oldest bookmark across all streams
        # messages between global_start_lsn and a stream's individual start_lsn
        # will be read from the WAL but filtered out of that stream's buffer
        global_start_lsn = self.get_flush_lsn()

        # comma-separated list of schema.table with special chars escaped
        add_tables = ",".join(
            _escape_wal2json_table_fqn(stream_cfg.table_fqn)
            for stream_cfg in self._streams.values()
        )

        conn = psycopg2.connect(
            self._connection_parameters.render_as_psycopg2_dsn(),
            connection_factory=extras.LogicalReplicationConnection,
        )
        cur = conn.cursor()
        try:
            # flush WAL up to the global start point using `send_feedback()``
            cur.send_feedback(flush_lsn=global_start_lsn)
            cur.start_replication(
                slot_name=self._replication_slot_name,
                decode=True,
                start_lsn=global_start_lsn,
                status_interval=self._status_interval,
                options={
                    "format-version": 2,
                    "include-transaction": False,
                    "add-tables": add_tables,
                },
            )

            # read loop
            while True:
                message = cur.read_message()

                if message:
                    self._handle_message(message, cur)
                else:
                    # no message available right now...
                    # wait up to the remaining status_interval for new data to arrive
                    elapsed = (datetime.now() - cur.feedback_timestamp).total_seconds()
                    remaining = self._status_interval - elapsed

                    try:
                        ready = select_mod.select([cur], [], [], max(0, remaining))[0]
                    except InterruptedError:
                        continue

                    if not ready:
                        # timeout expired with no new messages => finish
                        break
        finally:
            cur.close()
            conn.close()
            self._has_run = True

    def _handle_message(self, message, cursor) -> None:
        """Parse a WAL message and route it to the correct buffer(s).

        Args:
            message: ``psycopg2`` replication message, .e. has .payload, .data_start attributes
            cursor: live replication cursor, needed for text[] pre-parsing via STRINGARRAY
        """
        payload = _parse_wal_payload(message.payload, cursor)
        if payload is None:
            # unparseable JSON even after enum-quote fix: skip it
            return

        action = payload.get("action")
        if action in _SKIP_ACTIONS:
            # Begin/Commit/Truncate not routable to a specific stream buffer: skip it
            return

        # determine which table this message belongs to
        schema = payload.get("schema", "")
        table = payload.get("table", "")
        table_fqn = f"{schema}.{table}"
        # only buffer if this table was configured as a stream at init
        if table_fqn not in self._buffers:
            return

        lsn = message.data_start
        # filter: skip messages that predate this stream's bookmark
        stream_cfg = self._streams[table_fqn]
        if lsn < stream_cfg.start_lsn:
            return

        self._buffers[table_fqn].append(
            WALMessage(
                table_fqn=table_fqn,
                action=action,  # type: ignore[arg-type]
                lsn=lsn,
                payload=payload,
            )
        )

    def get_messages(self, table_fqn: str) -> list[WALMessage]:
        """Return prefetched messages for a given stream.

        Called by ``PostgresLogBasedStream.get_records()``.
        Returns a list (not a generator) because the WAL has already been fully read.

        Args:
            table_fqn: fully-qualified table name ("schema.table")

        Returns:
            List of :class:`WALMessage` objects, in LSN order;
            an empty list if the table had no WAL messages.

        Raises:
            RuntimeError: If :meth:`run()` has not been called yet.
            KeyError: If ``table_fqn`` was not registered at init time.
        """
        if not self._has_run:
            raise RuntimeError("WALPrefetcher.get_messages() called before run()")

        if table_fqn not in self._buffers:
            raise KeyError(
                f"Table {table_fqn!r} was not registered with this WALPrefetcher. "
                f"Registered tables: {sorted(self._buffers.keys())}"
            )
        return self._buffers[table_fqn]

    def get_flush_lsn(self) -> int:
        """Get minimum start_lsn across all registered streams.

        This is the safe point up to which Postgres can discard WAL -- every stream
        has already consumed everything before this LSN in a prior run.
        """
        return min(s.start_lsn for s in self._streams.values())

    def consume(self, payload: dict, lsn: int) -> dict | None:
        """Build a Singer-compatible row dict from a parsed ``wal2json`` payload.

        Args:
            payload: A parsed wal2json dict
                with text[] values already pre-parsed into Python lists.
            lsn: The WAL LSN for this message.

        Returns:
            A dict suitable for yielding from get_records(), or None/{}
            for non-data messages (truncate, transaction begin/commit).
        """
        row: dict = {}
        if payload["action"] in _UPSERT_ACTIONS:
            row = {
                column["name"]: self._parse_column_value(column) for column in payload["columns"]
            }
            row["_sdc_deleted_at"] = None
            row["_sdc_lsn"] = lsn
        elif payload["action"] in _DELETE_ACTIONS:
            row = {
                column["name"]: self._parse_column_value(column) for column in payload["identity"]
            }
            row["_sdc_deleted_at"] = _now_utc()
            row["_sdc_lsn"] = lsn
        else:
            raise RuntimeError(
                f"consume() received unexpected action {payload['action']!r}. "
                f"Only I/U/D expected; B/C/T should be filtered by _handle_message()."
            )

        return row

    def _parse_column_value(self, column):
        """Parse a single column value from a ``wal2json`` column dict.

        Handles ``text[]`` arrays -- expected to be pre-parsed into Python lists
        by the caller or prefetcher -- null values, and empty-string numerics.
        """
        value = column.get("value")
        if value is None:
            return None

        col_type = column.get("type", "")
        # for text[] arrays: if value is already a list, return as-is
        # if it's still a string (via legacy code path), handle it gracefully
        if col_type == "text[]":
            return (
                value
                # good: value has been pre-parsed
                if isinstance(value, list)
                # fallback: attempt to parse without cursor, handling the case
                # legacy code calls without pre-parsing... note: UTF8-only
                else STRINGARRAY(value, None)
            )

        # for numeric types: empty string should be treated as null
        if value == "" and col_type in _NUMERIC_DTYPES:
            return None

        return value


def _fix_wal2json_enum_quotes(payload: str) -> str:
    """Fix malformed JSON from ``wal2json`` for PostgreSQL enum types."""
    return _WAL2JSON_ENUM_QUOTE_RE.sub(r'"type":"\1"', payload)


def _pre_parse_text_arrays(payload: dict, cursor) -> dict:
    """Pre-parse text[] column values in a ``wal2json`` payload.

    Converts column values in-place, from text[] string representations into lists
    using ``cursor`` for encoding context. This gets called during WAL reading
    so that downstream ``consume()`` never needs a cursor.
    """
    # NOTE: wal2json format-version-2 uses "columns" for I/U and "identity" for D
    columns = payload.get("columns", payload.get("identity", []))
    for column in columns:
        if column.get("type") == "text[]" and column.get("value") is not None:
            column["value"] = STRINGARRAY(column["value"], cursor)
    return payload


def _parse_wal_payload(raw_payload: str, cursor) -> dict | None:
    """Parse a raw ``wal2json`` JSON string into a dict.

    Handles the enum quoting bug and pre-parses text[] values while the cursor is alive.
    Returns None if the payload can't be decoded.
    """
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError:
        fixed = _fix_wal2json_enum_quotes(raw_payload)
        try:
            payload = json.loads(fixed)
        except json.JSONDecodeError:
            return None

    _pre_parse_text_arrays(payload, cursor)
    return payload


def _escape_wal2json_table_fqn(fqn: str) -> str:
    """Escape a fully-qualified table name for wal2json's "add-tables" option.

    Expects ``fqn`` in "schema.table" format -- the period between schema and table
    must NOT be escaped. Special characters (space, single quote, comma, period, asterisk)
    within schema or table components are backslash-escaped.
    """
    schema, _, table = fqn.partition(".")
    for char in _WAL2JSON_CHARS_TO_ESCAPE:
        schema = schema.replace(char, f"\\{char}")
        table = table.replace(char, f"\\{char}")
    return f"{schema}.{table}"


def _now_utc() -> str:
    """Return the current UTC time as a string."""
    # TODO: duplicated from client.py; consider consolidating
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

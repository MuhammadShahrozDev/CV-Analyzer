"""Server-side state for a batch run.

Generation of a hundred emails takes minutes, so the browser cannot hold the
work in one request. The client asks for a chunk at a time and the server keeps
the expensive pieces - the parsed CV, the job rows, the results so far -
between those calls.

Kept in memory deliberately. A batch is a single operator working through one
workbook in one sitting; persisting it would mean a database, migrations and
cleanup for state that is worthless ten minutes later. The trade-off is that a
server restart loses an in-progress batch, which is why sessions are also
capped and aged out rather than growing without limit.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

# A batch older than this is abandoned. Long enough for a hundred-row run plus
# review time, short enough that a forgotten session does not hold a parsed CV
# in memory indefinitely.
SESSION_TTL_SECONDS = 2 * 60 * 60

# Guards against unbounded growth if sessions are created and never finished.
MAX_SESSIONS = 12


@dataclass
class BatchSession:
    """One operator's batch: the CV, the rows, and what has been generated."""

    session_id: str
    created_at: float
    profile: Any                          # CandidateProfile
    workbook_path: str
    sheet: str
    header_row: int
    mapping: dict[str, int]
    jobs: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    mode: str = "individual"
    demo_recipient: str = ""
    use_row_email: bool = False
    cv_filename: str = ""
    workbook_name: str = ""

    @property
    def generated_count(self) -> int:
        return sum(1 for item in self.results if item.get("status") == "generated")

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "cv_filename": self.cv_filename,
            "workbook_name": self.workbook_name,
            "total_rows": len(self.jobs),
            "processed": len(self.results),
            "generated": self.generated_count,
            "skipped": sum(1 for item in self.results if item.get("status") == "skipped"),
            "failed": sum(1 for item in self.results if item.get("status") == "failed"),
            "mode": self.mode,
        }


class BatchStore:
    """Thread-safe session registry.

    Uvicorn serves requests concurrently, so chunk requests for the same batch
    can overlap. A lock around mutation keeps the results list consistent.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, BatchSession] = {}
        self._lock = threading.Lock()

    def create(self, **kwargs: Any) -> BatchSession:
        session = BatchSession(
            session_id=uuid.uuid4().hex, created_at=time.time(), **kwargs
        )
        with self._lock:
            self._prune_locked()
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> Optional[BatchSession]:
        with self._lock:
            self._prune_locked()
            return self._sessions.get(session_id)

    def drop(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _prune_locked(self) -> None:
        """Remove expired sessions, then the oldest if still over the cap."""
        cutoff = time.time() - SESSION_TTL_SECONDS
        for key in [k for k, s in self._sessions.items() if s.created_at < cutoff]:
            self._sessions.pop(key, None)

        while len(self._sessions) > MAX_SESSIONS:
            oldest = min(self._sessions.values(), key=lambda s: s.created_at)
            self._sessions.pop(oldest.session_id, None)


store = BatchStore()
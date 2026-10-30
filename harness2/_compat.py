# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The two POSIX-only primitives the method uses, with a Windows fallback for each.

Everything else in core is portable; these exist so the DSH substrate can run on a
Windows host without forking the call sites' logic.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time

IS_WINDOWS = os.name == "nt"

# Windows has no SIGKILL; kill_group ignores the signal there and kills the tree outright.
SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)


def lock_exclusive(fh) -> None:
    """Block until `fh` holds an exclusive lock; released when `fh` is closed."""
    if not IS_WINDOWS:
        import fcntl

        fcntl.flock(fh, fcntl.LOCK_EX)
        return
    import msvcrt

    fh.seek(0)
    while True:
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK, 1)
            return
        except OSError:
            # LK_LOCK gives up after ten one-second retries; keep waiting like flock does.
            time.sleep(0.5)


def unlock(fh) -> None:
    """Release a lock taken with `lock_exclusive`."""
    if not IS_WINDOWS:
        import fcntl

        fcntl.flock(fh, fcntl.LOCK_UN)
        return
    import msvcrt

    fh.seek(0)
    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)


def kill_group(proc: subprocess.Popen, sig: int = signal.SIGTERM) -> None:
    """Signal the process group `proc` leads; on Windows, kill its whole process tree."""
    if not IS_WINDOWS:
        os.killpg(proc.pid, sig)
        return
    subprocess.run(
        ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


__all__ = ["IS_WINDOWS", "SIGKILL", "lock_exclusive", "unlock", "kill_group"]

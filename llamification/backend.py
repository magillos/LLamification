"""Managed subprocess for a provider's optional "start command".

A provider may define an arbitrary shell command line (for example
``llama-server -m model.gguf --port 8080`` or ``ollama serve``) that
LLamification runs before the proxy starts, so a local backend is already
listening by the time the first request arrives.

This module is UI-free so the GUI and the headless entry point share one
lifecycle implementation:

* output is forwarded line by line to the ``llamification.backend`` logger,
  which the GUI log pane and the headless terminal both display;
* the command runs in its own process group, so stopping it also stops the
  backend behind a ``sh -c`` wrapper;
* one :class:`BackendCommand` object can be kept across proxy stop/start
  cycles — :attr:`BackendCommand.is_running` tells the caller whether the
  command must be spawned again.
"""

import logging
import os
import signal
import socket
import subprocess
import threading
import time
from typing import Callable, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger("llamification.backend")

# How long the proxy waits for a local backend to accept connections before
# starting anyway. Loading a large model can take a while, but startup is
# never blocked longer than this.
READY_TIMEOUT = 60.0

_POLL_INTERVAL = 0.25
_TERMINATE_TIMEOUT = 5.0


def parse_host_port(url: str) -> Optional[Tuple[str, int]]:
    """Extract ``(host, port)`` from a provider base URL.

    Returns ``None`` when no usable host can be parsed. A missing port falls
    back to the scheme's default (80 for http, 443 for https).
    """
    url = (url or "").strip()
    if not url:
        return None
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = (parsed.hostname or "").strip()
    if not host:
        return None
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return None
    return host, int(port)


def is_loopback_host(host: str) -> bool:
    """True for addresses that can only refer to this machine."""
    host = (host or "").strip("[]").lower()
    return host == "localhost" or host == "::1" or host.startswith("127.")


class BackendCommand:
    """A provider's start command running as a child process."""

    def __init__(
        self,
        command: str,
        provider_name: str = "",
        terminate_on_stop: bool = False,
    ):
        self.command = command
        self.provider_name = provider_name or "backend"
        self.terminate_on_stop = terminate_on_stop
        self.exit_code: Optional[int] = None
        self._proc: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self._stopping = False

    # --- state -------------------------------------------------------

    @property
    def pid(self) -> Optional[int]:
        proc = self._proc
        return proc.pid if proc is not None else None

    @property
    def is_running(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    # --- lifecycle ---------------------------------------------------

    def start(self) -> bool:
        """Spawn the command. Returns ``False`` when it could not be started."""
        if self.is_running:
            return True
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            # Own session (and therefore own process group) so stop() can
            # signal the whole `sh -c` tree, not just the shell.
            kwargs["start_new_session"] = True
        try:
            proc = subprocess.Popen(
                self.command,
                shell=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                **kwargs,
            )
        except (OSError, ValueError) as exc:
            logger.error(
                "✗ Could not run start command for %s: %s", self.provider_name, exc
            )
            return False

        self._proc = proc
        self._stopping = False
        self.exit_code = None
        logger.info(
            "▶ Start command for %s (pid %s): %s",
            self.provider_name, proc.pid, self.command,
        )
        self._reader = threading.Thread(
            target=self._pump_output,
            args=(proc,),
            name="llamification-backend",
            daemon=True,
        )
        self._reader.start()
        return True

    def _pump_output(self, proc: subprocess.Popen) -> None:
        """Forward the command's output to the log, then record its exit."""
        stream = proc.stdout
        try:
            if stream is not None:
                for line in stream:
                    logger.info("[start] %s", line.rstrip("\r\n"))
        except (ValueError, OSError):
            pass  # pipe closed while stopping
        finally:
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
            code = proc.wait()
            self.exit_code = code
            if self._stopping:
                logger.debug(
                    "Start command for %s exited with code %s",
                    self.provider_name, code,
                )
            else:
                logger.info(
                    "■ Start command for %s exited with code %s",
                    self.provider_name, code,
                )

    def wait_ready(
        self,
        url: str,
        timeout: float = READY_TIMEOUT,
        should_abort: Optional[Callable[[], bool]] = None,
    ) -> bool:
        """Wait until the provider's host/port accepts connections.

        Only loopback URLs are waited for: a remote provider never delays
        startup because of someone else's network. Returns ``True`` once a
        connection succeeds, ``False`` on timeout, early exit or abort.
        """
        target = parse_host_port(url)
        if target is None or not is_loopback_host(target[0]):
            return False
        host, port = target
        deadline = time.monotonic() + timeout
        logger.info(
            "Waiting for %s:%s to accept connections (up to %.0fs)…",
            host, port, timeout,
        )
        while time.monotonic() < deadline:
            if should_abort is not None and should_abort():
                logger.info("Stopped waiting for %s:%s.", host, port)
                return False
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    logger.info("✓ %s:%s is accepting connections.", host, port)
                    return True
            except OSError:
                pass
            if not self.is_running:
                logger.info(
                    "Start command finished; %s:%s is not reachable — continuing.",
                    host, port,
                )
                return False
            time.sleep(_POLL_INTERVAL)
        logger.warning(
            "⚠ %s:%s did not accept connections within %.0fs; starting anyway.",
            host, port, timeout,
        )
        return False

    def stop(self, timeout: float = _TERMINATE_TIMEOUT) -> bool:
        """Terminate the command's process group.

        Returns ``False`` when nothing was running.
        """
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return False
        self._stopping = True
        pid = proc.pid
        self._signal(signal.SIGTERM)
        try:
            proc.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            logger.warning("Start command (pid %s) ignored SIGTERM; killing it.", pid)
        self._signal(signal.SIGKILL)
        try:
            proc.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            logger.error("✗ Could not stop start command (pid %s).", pid)
            return False

    def _signal(self, sig: int) -> None:
        """Send ``sig`` to the command's process group when possible."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
            )
            return
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except OSError:
            try:
                proc.send_signal(sig)
            except OSError:
                pass

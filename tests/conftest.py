import os
import sys

import pytest

_exit_status = 0


def pytest_sessionfinish(session, exitstatus):
    global _exit_status
    _exit_status = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    # On Linux, native libraries (ONNX Runtime / Chroma) can abort during interpreter teardown
    # after every test has passed (exit 134). The results are final by now, so exit directly.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_exit_status)

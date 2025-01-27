import re
from contextlib import contextmanager
from functools import partial
from sys import stderr
from warnings import catch_warnings, filterwarnings

err = partial(print, file=stderr)


@contextmanager
def suppress_datapipes_deprecation_warning():
    with catch_warnings():
        filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"torchdata\.datapipes",
            # Multi-line message matching doesn't seem to work
            # message=re.compile(".*datapipes.*", re.MULTILINE),
            # message=".*The 'datapipes', 'dataloader2' modules are deprecated.*",
        )
        yield

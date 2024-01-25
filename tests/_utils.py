"""Testing utils."""

from click.testing import CliRunner


def run_click_script(func, args: list[str], catch_exceptions: bool = False):
    """Util to test click scripts while showing the stdout."""

    runner = CliRunner()

    # We catch the exception here no matter what, but we'll reraise later if need be.
    result = runner.invoke(func, args, catch_exceptions=True)

    # Without this the output to stdout/stderr is grabbed by click's test runner.
    # print(result.output)

    # In case of an exception, raise it so that the test fails with the exception.
    if result.exception and not catch_exceptions:
        raise result.exception

    return result

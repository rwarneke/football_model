### Download latest match result data from martj42/international_results. ###

from pathlib import Path
import ssl
from urllib import error, request

BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master/"
FILES = (
    "results.csv",
    "shootouts.csv",
    "goalscorers.csv",
    "former_names.csv",
)


def _ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    try:
        import certifi
    except Exception:
        return ctx
    try:
        ctx.load_verify_locations(certifi.where())
    except Exception:
        return ctx
    return ctx


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with request.urlopen(url, context=_ssl_context()) as response, open(
            tmp, "wb"
        ) as f:
            f.write(response.read())
    except error.URLError as exc:
        if isinstance(exc.reason, ssl.SSLCertVerificationError):
            raise RuntimeError(
                "SSL verification failed. Try `python -m pip install certifi` or "
                "install system certificates for your Python."
            ) from exc
        raise
    tmp.replace(dest)


def main() -> None:
    target_dir = Path(__file__).resolve().parent.parent
    for name in FILES:
        url = f"{BASE_URL}{name}"
        dest = target_dir / name
        download(url, dest)
        print(f"updated {dest}")


if __name__ == "__main__":
    main()

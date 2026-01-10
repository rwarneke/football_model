### Download latest match result data from martj42/international_results. ###

from pathlib import Path
from urllib import request

BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master/"
FILES = (
    "results.csv",
    "shootouts.csv",
    "goalscorers.csv",
    "former_names.csv",
)


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with request.urlopen(url) as response, open(tmp, "wb") as f:
        f.write(response.read())
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

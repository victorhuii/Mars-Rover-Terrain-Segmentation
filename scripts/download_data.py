import argparse
from pathlib import Path
import urllib.request

from audit_dataset import FILES, REVISION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('data/raw'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        destination = args.output / filename
        if destination.exists():
            print(f'Using existing {destination}', flush=True)
            continue
        url = f'https://huggingface.co/datasets/hassanjbara/AI4MARS/resolve/{REVISION}/data/{filename}'
        temporary = destination.with_suffix(destination.suffix + '.part')
        print(f'Downloading {filename}', flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as target:
            while chunk := response.read(4 * 1024 * 1024):
                target.write(chunk)
        temporary.replace(destination)


if __name__ == '__main__':
    main()

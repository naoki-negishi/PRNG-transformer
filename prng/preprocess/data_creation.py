import json
import argparse
from pathlib import Path
from typing import Optional
from collections.abc import Iterator


class LinearCongruentialGenerator:
    def __init__(
        self,
        a: int = 33,
        b: int = 0,
        m: int = 2**8 - 5,
    ) -> None:
        self.seed: Optional[int] = None
        self.a = a
        self.b = b
        self.m = m

    def next(self) -> int:
        assert self.seed is not None, "Seed is not set"
        self.seed = (self.a * self.seed + self.b) % self.m
        return self.seed

    def set_seed(self, seed: int) -> None:
        self.seed = seed


def create_loader(
    algorithm: str
) -> tuple[Iterator[int, int, None], dict[str, int]]:
    if algorithm == "lcg":
        generator = LinearCongruentialGenerator()
        params_dict = {"a": generator.a, "b": generator.b, "m": generator.m}
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return generator, params_dict


def create_dataset(
    generator: Iterator[int, int, None],
    num_seeds: int,
    iter_per_seed: int = 512
) -> tuple[int, list[int]]:
    # try seed=0, seed=1, ..., seed=num_seeds-1
    for seed in range(1, num_seeds+1):
        generator.set_seed(seed)
        num_seq = []
        for _ in range(iter_per_seed):
            num_seq.append(generator.next())
        yield seed, num_seq


def main(args: argparse.Namespace):
    output_path = args.output_path
    Path(output_path).mkdir(parents=True, exist_ok=True)

    generator, params_dict = create_loader(args.alg)
    params = ''.join(f'{k}={v}' for k, v in params_dict.items())
    with open(output_path + f"/{args.alg}_{params}" + '.jsonl', "w") as f:
        for seed, num_seq in create_dataset(generator, args.num_seeds, args.iter_per_seed):
            data = {"seed": seed, "num_seq": num_seq}
            json.dump(data, f)
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_path", type=str)
    parser.add_argument("--alg", type=str, default="lcg")
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=100,
        help="How may seeds to try"
    )
    parser.add_argument(
        "--iter_per_seed",
        type=int,
        default=512,
        help="How many iterations per one seed"
    )
    args = parser.parse_args()
    main(args)

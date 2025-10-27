import argparse
from .core import ScenarioIntegrator


def main():
ap = argparse.ArgumentParser(description="Scenario Integration – quick CLI")
sub = ap.add_subparsers(dest="cmd", required=True)


p1 = sub.add_parser("process")
p1.add_argument("year", type=int)
p1.add_argument("scenario", type=str)
p1.add_argument("--out", required=True)


p2 = sub.add_parser("dc-add")
p2.add_argument("year", type=int)
p2.add_argument("base_year", type=int)
p2.add_argument("--adding", type=float, default=100000)
p2.add_argument("--out", required=True)


args = ap.parse_args()
si = ScenarioIntegrator()


if args.cmd == "process":
si.process_to_csv(out_path=args.out, year=args.year, scenario=args.scenario)
elif args.cmd == "dc-add":
si.add_datacenter_capacity_to_csv(out_path=args.out, year=args.year, base_year=args.base_year, adding_MW=args.adding)


if __name__ == "__main__":
main()
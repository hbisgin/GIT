import argparse

parser = argparse.ArgumentParser("test out")

parser.add_argument("--gpu")
parser.add_argument("protocol")

args = parser.parse_args()

print(args.gpu)
print(args.protocol)

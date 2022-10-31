from pathlib import Path

csv_path =Path("/home/user/halil/master-suter/data/csv/")
protocol = 1

for v in sorted(csv_path.glob(f'*{protocol}*')):
        print(v)
        print(str(Path(v.name).stem).replace(f'_p{protocol}', '').replace(f'p{protocol}', ''))

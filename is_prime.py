import os
from pathlib import Path
import pandas as pd
path_to_this_file = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(str(Path(path_to_this_file) / "primes.txt"))

def is_prime(value):
  return any(df.first_million_primes == value)

if __name__ == "__main__":
  print(is_prime(57))

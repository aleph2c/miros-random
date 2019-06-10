import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path('.') / '..'))
from random_number import width_key
from random_number import depth_key
from random_number import period_key
from random_number import repeat_key
from random_number import angle_key
from random_number import periodicity
from random_number import automata_periodicity
from random_number import OneDCellularAutonomataWallRecursion

specifications = []
for width in range(10, 23):
  machine_specifications = automata_periodicity(width=width, start_depth=2, stop_depth=31)
  unfiltered_as_list = [machine_specifications[depth] for depth in range(2, 31)]
  specifications += [
    candidate for candidate in unfiltered_as_list if
    candidate['repeat?'] == True
  ]

generations = 1
machines = [OneDCellularAutonomataWallRecursion(
  cells_per_generation=spec[width_key],
  queue_depth=spec[depth_key],
  generations=generations) for spec in specifications
  ]

def size_of_com(n, r):
  # we need to make sure that a new item is 
  # used in every combination of our mixing set
  n = n - 1
  r = r - 1
  # n!/((n-r)!*r!)
  def fact(z):
    _fact = 1
    for i in range(1, z+1):
      _fact += _fact * i
    return _fact
  num = fact(n)
  den = fact(n-r)
  den *= fact(r)
  if den:
    ans = num/den
  else:
    ans = 0
  return ans

# if you mix 62 machines in a table of 250
# we are talking 10**57 (big enough to defeat quantum computers)
number_of_machines = 250
for i in range(3, number_of_machines):
  print(size_of_com(number_of_machines, i), i)




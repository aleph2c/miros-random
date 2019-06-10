
import os
import sys
import pytest
import subprocess
from random_number import Canvas
from random_number import Rule30
from random_number import WallLeftWhiteRightWhite
from random_number import OneDCellularAutonomataWallRecursion
from random_number import ODCAWRPeriodDetection
from random_number import automata_periodicity

# pytest is too stupid to find files
path_to_this_file = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path_to_this_file + '/../')

# To Look:
# ma = OneDCellularAutonomataWallRecursion(
#   machine_cls=Rule30,
#   wall_cls=WallLeftWhiteRightWhite,
#   cells_per_generation=width,
#   queue_depth=depth,
#   generations=generations,
# )
# eco = Canvas(ma)
# eco.run_animation(generations, interval=100)
# filename = "no_looping_duration_discovery"
# pdf = '{}.pdf'.format(filename)
# eco.save(pdf)
# cmd = 'cmd.exe /C {} &'.format('{}.pdf'.format(filename))
# subprocess.Popen(cmd, shell=True)

@pytest.mark.twentytwo
def test_period_discovery_with_mask():
  # we know rule 22 points at which it stops
  # add a mask which will detect for this
  result = automata_periodicity(22, 6)
  assert(result[6]['repeat?'] == False)
  assert(result[6]['period'] == 195)
  assert(result[6]['width_of_automata'] == 22)
  result = automata_periodicity(22, 7)
  assert(result[7]['repeat?'] == False)
  assert(result[7]['period'] == 133)
  assert(result[7]['width_of_automata'] == 22)
  result = automata_periodicity(22, 8)
  assert(result[8]['repeat?'] == False)
  assert(result[8]['period'] == 167)
  assert(result[8]['width_of_automata'] == 22)

@pytest.mark.twentytwo
def test_period_discovery_without_mask():
  result = automata_periodicity(22, 2)
  assert(result[2]['repeat?'] == True)
  assert(result[2]['period'] == 113)
  assert(result[2]['width_of_automata'] == 22)

@pytest.mark.long
def test_period_discovery_long_running():
  for width in range(27, 28):
    automata_periodicity(width, 16, 30)


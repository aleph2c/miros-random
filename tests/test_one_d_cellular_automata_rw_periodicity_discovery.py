
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

@pytest.mark.twentytwo
def test_period_discovery_with_mask():
  # we know rule 22 points at which it stops
  # add a mask which will detect for this
  # 22, 6, 197, False
  # 22, 7, 135, False
  # 22, 8, 169, False

  generations = 133
  width = 22
  depth = 8
  ma = OneDCellularAutonomataWallRecursion(
    machine_cls=Rule30,
    wall_cls=WallLeftWhiteRightWhite,
    cells_per_generation=width,
    queue_depth=depth,
    generations=generations,
  )
  eco = Canvas(ma)
  eco.run_animation(generations, interval=100)
  filename = "no_looping_duration_discovery"
  pdf = '{}.pdf'.format(filename)
  eco.save(pdf)
  cmd = 'cmd.exe /C {} &'.format('{}.pdf'.format(filename))
  subprocess.Popen(cmd, shell=True)

  period = automata_periodicity(22, depth)

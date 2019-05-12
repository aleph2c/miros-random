
import os
import sys
import pytest
import subprocess

# pytest is too stupid to find files
path_to_this_file = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path_to_this_file + '/../')

from random_number import Rule30
from random_number import Canvas 
from random_number import WallLeftWhiteRightWhite
from random_number import OneDCellularAutomataWithAngleDiscoveryAtMiddle 
#OneDCellularAutomataWithAngleDiscoveryAtMiddle

@pytest.mark.angle
def test_angle_discovery():
  OneDCellularAutomataWithAngleDiscoveryAtMiddle
  generations = 100
  width = 30
  ma = OneDCellularAutomataWithAngleDiscoveryAtMiddle(
    generations=generations,
    machine_cls=Rule30,
    wall_cls=WallLeftWhiteRightWhite,
    cells_per_generation=width
    )
  eco = Canvas(ma)
  eco.run_animation(generations, interval=100)
  filename = "test_angle_discovery"
  pdf = '{}.pdf'.format(filename)
  eco.save(pdf)
  cmd = 'cmd.exe /C {} &'.format('{}.pdf'.format(filename))
  subprocess.Popen(cmd, shell=True)
  print(ma.n_angle)

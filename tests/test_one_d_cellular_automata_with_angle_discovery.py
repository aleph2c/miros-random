
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
#from random_number import OneDCellularAutomataWithAngleDiscovery
from random_number import OneDCellularAutomataWithAngleDiscoveryAtMiddle 

from random_number import AngleAndDepthDiscovery
from random_number import angle_and_depth

@pytest.mark.angle
def test_angle_discovery():
  generations = 100
  width = 12
  ma = OneDCellularAutomataWithAngleDiscoveryAtMiddle(
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
  print(ma.nothing_at_row)

  ma = OneDCellularAutomataWithAngleDiscoveryAtMiddle(
    machine_cls=Rule30,
    wall_cls=WallLeftWhiteRightWhite,
    cells_per_generation=width
  )
  discovery = AngleAndDepthDiscovery(ma)
  print(discovery.n_angle)
  print(discovery.queue_depth)

@pytest.mark.angle
@pytest.mark.depth
def test_angle_and_depth_discover_6_to_9():
  results = angle_and_depth(start_width=6, stop_width=9)
  print(results)
  with pytest.raises(IndexError):
    results[5]
  with pytest.raises(IndexError):
    results[9]
  assert(int(results[6]['angle_of_n_phenomenon_on_left']) == 56)
  assert(int(results[6]['queue_depth']) == 5.0)
  assert(int(results[6]['width_of_automata']) == 6.0)

@pytest.mark.angle
@pytest.mark.depth
def test_angle_and_depth_discover_14():
  results = angle_and_depth(start_width=14)
  print(results)

@pytest.mark.depth
def test_going_big():
  results = angle_and_depth(start_width=6, stop_width=485)

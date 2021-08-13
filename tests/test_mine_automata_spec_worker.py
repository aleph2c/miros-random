import os
import sys
import time
import pytest
import subprocess
from miros import Event
from miros import signals
from miros import Factory
from collections import deque
from miros import return_status
from random_number import Canvas
from random_number import Rule30
from random_number import WallLeftWhiteRightWhite
from random_number import OneDCellularAutonomataWallRecursion
from random_number import ODCAWRPeriodDetection
from random_number import automata_periodicity
from random_number import MineAutomataSpecWorker

# pytest is too stupid to find files
path_to_this_file = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path_to_this_file + '/../')

class MineAutomataSpecWorkerCatcher(Factory):

  def __init__(self, name=None):

    super().__init__(name)
    self.buffer = deque(maxlen=1)

    # build the statechart structure
    self.wait_for_published_event = self.create(state="wait_for_published_event"). \
      catch(signal=signals.Pioneer_Complete,
        handler=self.wait_for_published_pioneer_complete). \
      to_method()
    self.nest(self.wait_for_published_event, parent=None)

    # subscribe to the event we want to confirm is posted
    self.subscribe(Event(signal=signals.Pioneer_Complete))

    # startup this statechart and pause for its thread to stabilize
    self.start_at(self.wait_for_published_event)
    time.sleep(0.01)

  @staticmethod
  def wait_for_published_pioneer_complete(catcher, e):
    catcher.buffer.append(e)    
    return return_status.HANDLED

  def get_published_item(self):
    return self.buffer.popleft() if len(self.buffer) != 0 else None

@pytest.mark.worker
def test_mine_automata_spec_worker_ctor():
  catcher = MineAutomataSpecWorkerCatcher()
  worker = MineAutomataSpecWorker(cells_per_generation=10, queue_depth=10, name='bob')
  time.sleep(1.0)
  result = catcher.get_published_item()
  print(result)

@pytest.mark.worker
def test_mine_automata_spec_worker_stops():
  catcher = MineAutomataSpecWorkerCatcher()
  worker = MineAutomataSpecWorker(cells_per_generation=10, queue_depth=10, name='bob')
  time.sleep(1.0)
  result = catcher.get_published_item()
  time.sleep(1.0)
  print(result)
  assert(worker.thread.is_alive()==False)

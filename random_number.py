import os
import re
import csv
import math
import pathlib
import subprocess
import datetime as dt
from functools import reduce
from collections import deque
from collections import namedtuple
from collections import OrderedDict

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from miros import Event
from miros import signals
from miros import HsmWithQueues
from miros import return_status

import is_prime

White   = 0
Black   = 1

width_key = 'width_of_automata'
depth_key = 'queue_depth'
period_key = 'period'
repeat_key = 'repeat?'
angle_key = 'angle_of_n_phenomenon_on_left'

def autocorrelate(x):
  '''

  compute the autocorrelation of vector x

  **Args**:
     | ``x`` (list): a list containing where each element is a 1.0 or 0.0

  **Returns**:
     (list): the autocorrelation for the list
  '''
  result = np.correlate(x, x, mode='full')
  # don't include the correletion with itself
  result[result.size//2] = 0
  return result[result.size//2:]

def build_rule_function(number):
  if number > 2**8:
    print('not supported')
  number_as_binary_string = "{0:b}".format(number)
  number_as_binary_string = number_as_binary_string.rjust(8, '0')
  masks = {}
  masks[0] = number_as_binary_string[7] == '1'
  masks[1] = number_as_binary_string[6] == '1'
  masks[2] = number_as_binary_string[5] == '1'
  masks[3] = number_as_binary_string[4] == '1'
  masks[4] = number_as_binary_string[3] == '1'
  masks[5] = number_as_binary_string[2] == '1'
  masks[6] = number_as_binary_string[1] == '1'
  masks[7] = number_as_binary_string[0] == '1'
  def fn(left, middle, right):
    left_bit = 1 if left else 0
    middle_bit = 1 if middle else 0
    right_bit = 1 if right else 0
    number = left_bit << 2
    number += middle_bit << 1
    number += right_bit
    return masks[number]
  return fn

# credit agf from stackoverlow
# https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python 
def factors(n):
  return set(reduce(list.__add__,
    ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0 )))

def aggregate_period(num1, num2):
  num1 = int(num1)
  num2 = int(num2)
  product = num1 * num2
  divisor = max(factors(num1).intersection(factors(num2)))
  if divisor == 0:
    raise("0 illegal in period_lookup")
  return int(product/divisor)

def periodicity(*args):
  list_of_variables = args

  # contract
  def check_element(element):
    assert(hasattr(element, 'cells_per_generation')==True)
    assert(hasattr(element, 'queue_depth')==True)
  [check_element(element) for element in list_of_variables]

  def period_lookup(width, depth):
    result_as_list_list = automata_periodicity(width=width, start_depth=depth)
    period = result_as_list_list[depth]['period']
    return period

  unique_pattern_durations = [
    period_lookup(width=pattern.cells_per_generation, depth=pattern.queue_depth) for pattern in list_of_variables
  ]
  period = reduce(aggregate_period, unique_pattern_durations)
  return period

class Wall(HsmWithQueues):

  def __init__(self, name='wall'):
    super().__init__(name)
    self.color = None

  def color_number(self):
    return Black if self.color == 'black' else White

def fake_white(wall, e):
  status = return_status.UNHANDLED

  if(e.signal == signals.ENTRY_SIGNAL):
    wall.color = 'white'
    status = return_status.HANDLED
  elif(e.signal == signals.Next):
    status = return_status.HANDLED
  else:
    wall.temp.fun = wall.top
    status = return_status.SUPER
  return status

def fake_black(wall, e):
  status = return_status.UNHANDLED

  if(e.signal == signals.ENTRY_SIGNAL):
    wall.color = 'black'
    status = return_status.HANDLED
  elif(e.signal == signals.Next):
    status = return_status.HANDLED
  else:
    wall.temp.fun = wall.top
    status = return_status.SUPER
  return status

class WallLeftWhiteRightWhite(Wall):
  left_wall = fake_white
  right_wall = fake_white

class WallLeftWhiteRightBlack(Wall):
  left_wall = fake_white
  right_wall = fake_black

class WallLeftBlackRightWhite(Wall):
  left_wall = fake_black
  right_wall = fake_white

class WallLeftBlackRightBlack(Wall):
  left_wall = fake_black
  right_wall = fake_black

class Rule(Wall):
  def __init__(self, name='cell'):
    super().__init__(name)
    self.left = None
    self.right = None
    self.color = None

  def result(self, left, middle, right):
    inputs_as_bool = [True if cell.color == 'black' else False for cell in [left, middle, right]]
    output_as_bool = self.rule_fn(*inputs_as_bool)
    output_as_color = 'black' if output_as_bool else 'white'
    return output_as_color

class Rule30(Rule):

  def __init__(self, name='cell'):
    super().__init__(name)
    self.left = None
    self.right = None
    self.color = None
    self.rule_fn = build_rule_function(30)

def white(cell, e):
  status = return_status.UNHANDLED

  if(e.signal == signals.ENTRY_SIGNAL):
    cell.color = 'white'
    status = return_status.HANDLED
  elif(e.signal == signals.Next):
    next_color = cell.result(cell.left, cell, cell.right)
    if next_color == 'black':
      status = cell.trans(black)
    else:
      status = return_status.HANDLED
  else:
    cell.temp.fun = cell.top
    status = return_status.SUPER
  return status

def black(cell, e):
  status = return_status.UNHANDLED

  if(e.signal == signals.ENTRY_SIGNAL):
    cell.color = 'black'
    status = return_status.HANDLED
  elif(e.signal == signals.Next):
    next_color = cell.result(cell.left, cell, cell.right)
    if next_color == 'white':
      status = cell.trans(white)
    else:
      status = return_status.HANDLED
  else:
    cell.temp.fun = cell.top
    status = return_status.SUPER
  return status

class Rule30WithQueueDepth(Rule30):

  def __init__(self, name='cell'):
    super().__init__(name)

  @staticmethod
  def queue_depth(cells_per_generation): 

    n_cache = NCache()
    degrees = n_cache.get_angle(cells_per_generation)
    print("degrees {}".format(degrees))

    qd = 1 + math.tan(math.radians(degrees))
    qd *= 0.5*cells_per_generation
    qd = math.floor(qd)

    qd =  0.5 * math.tan(math.radians(degrees))
    qd += 1.0 / math.sqrt(2)
    qd *= cells_per_generation
    qd = math.floor(qd)
    #print(qd)
    #qd = 1
    return qd

class OneDCellularAutomata():
  def __init__(self,
      generations,
      cells_per_generation=None,
      initial_condition_index=None,
      machine_cls=None, 
      wall_cls=None,
      ):
    '''Build a two dimensional cellular automata object which can be advanced with a coroutine.  

    **Args**:
       | ``generations`` (int): how many generations to run (vertical cells)
       | ``cells_per_generation=None`` (int): how many cells across
       | ``initial_condition_index=None`` (int): the starting index cell (make black)
       | ``machine_cls=None`` (Rule): which automata rule to follow
       | ``wall_cls=None`` (Wall): which wall rules to follow

    **Returns**:
       (OneDCellularAutonomata): an automata object

    **Example(s)**:
      
    .. code-block:: python
     
      # build an automata using rule 30 with white walls
      # it should be 50 cells across
      # and it should run for 1000 generations
      ma = OneDCellularAutomata(
        machine_cls=Rule30,
        generations=1000,
        wall_cls=WallLeftWhiteRightWhite,
        cells_per_generation=50
      )

      # to get the generator for this automata
      generation = automata.make_generation_coroutine()

      # to advance a generation (first one will initialize it)
      next(generation)

      # to get the color codes from it's two dimension array
      automata.Z

      # to advance a generation
      next(generation)

    '''
    self.machine_cls = machine_cls
    self.wall_cls = wall_cls

    if machine_cls is None:
      self.machine_cls = Rule30

    if wall_cls is None:
      self.wall_cls = WallLeftWhiteRightWhite

    self.generations = generations + 1
    self.cells_per_generation = cells_per_generation

    # if they haven't specified cells_per_generation set it
    # so that the cells appear square on most terminals
    if cells_per_generation is None:
      # this number was discovered through trial and error
      # matplotlib seems to be ignoring the aspect ratio
      self.cells_per_generation = round(generations*17/12)

    self.initial_condition_index = round(self.cells_per_generation/2.0) \
      if initial_condition_index is None else initial_condition_index

    self.generation = None

    self.left_wall=self.wall_cls.left_wall
    self.right_wall=self.wall_cls.right_wall

  def make_and_start_left_wall_machine(self):
    '''make and start the left wall based on the wall_cls'''
    wall = self.wall_cls()
    wall.start_at(self.wall_cls.left_wall)
    return wall

  def make_and_start_right_wall_machine(self):
    '''make and start the right wall based on the wall_cls'''
    wall = self.wall_cls()
    wall.start_at(self.wall_cls.right_wall)
    return wall

  def initial_state(self):
    '''initialize the 2d cellular automata'''
    Z = np.full([self.generations, self.cells_per_generation], Black, dtype=np.bool)

    # create a collections of unstarted machines
    self.machines = []
    for i in range(self.cells_per_generation-2):
      self.machines.append(self.machine_cls())

    left_wall = self.make_and_start_left_wall_machine()
    right_wall = self.make_and_start_right_wall_machine()

    # unstarted machines sandwiched between unstarted boundaries
    self.machines = [left_wall] + self.machines + [right_wall]

    # start the boundaries in their holding color
    self.machines[0].start_at(fake_white)
    self.machines[-1].start_at(fake_white)

    # start most of the machines in white except for the one at the
    # intial_condition_index
    initial_bulk_colors = []
    for i in range(1, len(self.machines)-1):
      if i != self.initial_condition_index:
        self.machines[i].start_at(white)
        initial_bulk_colors.append(self.machines[i].color_number())
      else:
        self.machines[i].start_at(black)
        initial_bulk_colors.append(self.machines[i].color_number())
    Z[-1, 1:-1] = initial_bulk_colors
    # we have created a generation, so count down by one
    self.generation = self.generations-1

    ## draw our boundaries once, since they aren't going to change
    Z[:, 0] = self.machines[0].color_number()
    Z[:, Z.shape[-1]-1] = self.machines[-1].color_number()

    self.Z = Z

  def next_generation(self):
    '''create the next row of the 2d cellular automata, update the color map Z'''
    Z = self.Z
    if self.generation == self.generations-1:
      # draw the first row
      for i, machine in enumerate(self.machines):
        Z[self.generations-1, i] = machine.color_number()
    else:
      # draw every other row
      Z = self.Z
      new_machines = []
      for i in range(1, (len(self.machines)-1)):
        old_left_machine = self.machines[i-1]
        old_machine = self.machines[i]
        old_right_machine = self.machines[i+1]
        
        new_machine = self.machine_cls()
        new_machine.start_at(old_machine.state_fn)
        new_machine.left = old_left_machine
        new_machine.right = old_right_machine
        new_machines.append(new_machine)

      left_wall = self.make_and_start_left_wall_machine()
      right_wall = self.make_and_start_right_wall_machine()
      new_machines = [left_wall] + new_machines + [right_wall]

      for i, machine in enumerate(new_machines):
        machine.dispatch(Event(signal=signals.Next))
        Z[self.generation, i] = machine.color_number()
      self.machines = new_machines[:]

    self.Z = Z
    self.generation -= 1

  def make_generation_coroutine(self):
    '''create a coroutine, which can be used as many times needed'''
    self.initial_state()
    yield self.Z
    while True:
      self.next_generation()
      yield self.Z
      if self.generation == 0:
        raise RuntimeError

class OneDCellularAutomataWithAngleDiscoveryAtMiddle(OneDCellularAutomata):

  def __init__(self, 
        cells_per_generation=None, 
        initial_condition_index=None,
        machine_cls=None,
        wall_cls=None):
    '''An automata which will raise a RuntimeError when the n-phenonenon is
    discovered.  

    **Note**:
       This object is intended to be used by another object to provide the loop
       call to it's coroutine. (next)

    **Args**:
       | ``cells_per_generation=None`` (int): the width of the automata
       | ``initial_condition_index=None`` (int): where to place the first black
       |                                         cell in the first row of the automata
       | ``machine_cls=None`` (int): the 1-D cellular automata in the bulk
       | ``wall_cls=None`` (Wall): the type of wall class

    **Returns**:
       (OneDCellularAutomataWithAngleDiscoveryAtMiddle):  this object

    **Example(s)**:
      
    .. code-block:: python

        search_automata = OneDCellularAutomataWithAngleDiscoveryAtMiddle(
          machine_cls=Rule30,
          wall_cls=WallLeftWhiteRightWhite,
          cells_per_generation=width
        )
        discovery = AngleAndDepthDiscovery(search_automata)

        # The AngleAndDepthDiscovery consumes the
        # OneDCellularAutomataWithAngleDiscoveryAtMiddle, by running a loop
        # until the RuntimeError is detected.
        discovery = PeriodicityDiscovery(search_automata)
    '''

    depth_prior_to_n = int(cells_per_generation / 2)
    opposite = int(cells_per_generation / 2)
    angle_in_degrees = 23.0
    adjacent = int(opposite/math.tan(math.radians(angle_in_degrees)))
    generation_estimate = depth_prior_to_n + adjacent
    generations = generation_estimate * 2

    super().__init__(
      generations,
      cells_per_generation,
      initial_condition_index,
      machine_cls,
      wall_cls)

    self.black_mask = np.array([Black], dtype=np.float32)
    self.white_mask = np.array([White], dtype=np.float32)

    if self.wall_cls == WallLeftWhiteRightBlack or \
      self.wall_cls == WallLeftWhiteRightWhite:
      self.n_mask = np.concatenate(
         (self.white_mask, self.black_mask), axis=0)
    else:
      self.n_mask = np.concatenate(
         (self.black_mask, self.white_mask), axis=0)
    self.n_angle = 90

  def build_next_mask(self):

    # alternate color
    if self.n_mask[-1] == self.white_mask:
      self.n_mask = np.concatenate(
        (self.n_mask, self.black_mask), axis=0)
    else:
      self.n_mask = np.concatenate(
        (self.n_mask, self.white_mask), axis=0)

    # place the far right wall object based on wall type
    if len(self.n_mask) >= self.cells_per_generation:
      self.n_mask = self.n_mask[0:self.cells_per_generation]
      self.n_mask[-1] = self.white_mask if \
        self.wall_cls == WallLeftBlackRightWhite or WallLeftWhiteRightWhite else \
        self.black_mask

  def update_angle(self):
    previous_generation = self.generation+1
    row_to_check = self.Z[previous_generation]
    sub_row_to_check = row_to_check[0:len(self.n_mask)]

    # check up to the halfway point
    if np.array_equal(self.n_mask, sub_row_to_check):
      if len(self.n_mask) == self.initial_condition_index+1:
        self.nothing_at_row = self.generations-previous_generation
        adjacent = self.nothing_at_row
        adjacent -= (self.cells_per_generation / 2.0)
        opposite = self.cells_per_generation/2.0
        self.n_angle = math.degrees(math.atan(opposite/adjacent))
        raise RuntimeError

      self.build_next_mask()

  def next_generation(self):
    super().next_generation()
    self.update_angle()

class AngleAndDepthDiscovery():
  def __init__(self, automata):
    '''Iterates over an OneDCellularAutomataWithAngleDiscoveryAtMiddle object,
    which is provided as an argument.  When the
    OneDCellularAutomataWithAngleDiscoveryAtMiddle object's generator has found the
    conditions where an angle has been found it issues a RuntimeError exception
    (like all generators these days).

    This exception is caught by this object, the angle and the nothing_at_row
    information are added to this object's attributes and it returns.

    **Args**:
       | ``automata`` (OneDCellularAutomataWithAngleDiscoveryAtMiddle): an
       |              automata which will issue a RuntimeError exception when it has finished
       |              searching

    **Returns**:
       (AngleAndDepthDiscovery): an object with a angle and nothing_at_row attribute

    **Example(s)**:
      
    .. code-block:: python
       
        search_automata = OneDCellularAutomataWithAngleDiscoveryAtMiddle(
          machine_cls=Rule30,
          wall_cls=WallLeftWhiteRightWhite,
          cells_per_generation=20
        )
        discovery = AngleAndDepthDiscovery(search_automata)
        print(discovery.angle)  # => 21.03..
        print(discovery.nothing_at_row)  # => 36

    '''
    self.generation = automata.make_generation_coroutine()
    while(True):
      try:
        next(self.generation)
      except RuntimeError:
        break
    self.n_angle = automata.n_angle
    self.queue_depth = automata.nothing_at_row

class ListLike():
  def __init__(self, start_index, stop_index, _list):
    '''return a list like object, based on _list, which has been shifted
    start_index numbers to the right.

    **Note:**
      The resulting list only looks like it has been shifted to the right when
      it is being indexed.  This object is not actually a list, it does not
      support wrapping or slicing, you can only access its data using square
      brackets.

      The jury is out about if this is even a good idea, I may refactor away
      this class if it doesn't make things easier to read in the code.

    **Args**:
       | ``start_index`` (int): The starting index
       | ``stop_index`` (int): The stop index (not included in object)
       | ``_list`` (list): The list to shift to the right by start_index locations

    **Returns**:
       (ListLike): something that is easy to index into

    **Example(s)**:
      
    .. code-block:: python
       
       list_like = ListLike(8, 10, [8, 9])
       list_like[7]  # => IndexError exception
       list_list[8]  # => 8
       list_list[9]  # => 9
       list_list[10] # => IndexError exception

    '''
    self.start_index = start_index
    self.stop_index = stop_index 
    self.list = _list

  def __getitem__(self, key):
    if key < self.start_index:
      raise IndexError
    if key >= self.stop_index:
      raise IndexError
    index = key - self.start_index
    return self.list[index]

  def __repr__(self):
    return "start_index: {}, stop_index: {}, list: {}".format(
      self.start_index,
      self.stop_index, self.list)

def angle_and_depth(start_width, stop_width=None):
  '''Get the angle and central depth of a rule 30 cellular automata held in
  white walls.

  The angle [degrees] is a measure of the rule 30 bulk entropy angle as it is
  being consumed by the n-phenomenon at the bottom of the automata.  The depth
  [cells], is the number of cells down the center of the automata, before this
  column is consumed by the n-phenonenon.  See
  https://aleph2c.github.io/miros-random/cellular_automata.html for pictures
  details.

  **Note**:
     This function is memoized (cached) in the file named:
     'rule_30_with_white_walls_depth_and_angle.txt'.  This file is opened and
     updated with the panda python library.  Everytime a novel query is made the
     program speeds up and the cached file gets a bit larger.

     If a run of of an individual width/depth calculation takes 60 seconds or
     more, the cached file is saved immediately after the calculation is
     completed.  This way the user can stop midway through a long run and not
     loose the data that has been calculated up until that moment of
     cancellation.

  **Args**:
     | ``start_width`` (int): which width to start our query (included in result)
     | ``stop_width=None`` (int): which width to stop query (not included in result)

  **Returns**:
     (ListLike): a list like object that can be indexes with a width int to
     return diction information about that width.  The keys of the dictionaries
     are the same as the column names of the
     'rule_30_with_white_walls_depth_and_angle.txt' cached file.

  **Example(s)**:
    
  .. code-block:: python
     
     results = angle_and_depth(start_width=6, stop_width=9)

     print(results[5])  #=> IndexError exception

     print(results[6])  # =>
      {'angle_of_n_phenomenon_on_left': 56.309932474020215, 'width_of_automata':
      6.0, 'queue_depth': 5.0}
     print(results[7])  # => ...
     print(results[8])  # => ...

     print(results[9])  #=> IndexError exception

  '''
  if stop_width is None:
    stop_width = start_width + 1
  queue_depth = []
  results_dict = {}
  width_of_automata = []
  angle_of_n_phenomenon_on_left = []

  sample_times = deque(maxlen=2)
  sample_times.append(dt.datetime.now())
  sample_times.append(dt.datetime.now())

  column_titles = [
    width_key,
    angle_key,
    depth_key]
  filename = 'rule_30_with_white_walls_depth_and_angle.txt'

  try:
    df = pd.read_csv(filename)
    old_data = df.to_dict()
  except:
    old_data = {}
    df = pd.DataFrame({
      width_key: [],
      depth_key: [],
      angle_key: []})
    df = df[column_titles]

  def update_csv_file(old_data, width_of_automata, queue_depth, angle_of_n_phenomenon_on_left):
    new_data = {
      width_key: width_of_automata,
      depth_key: queue_depth,
      angle_key: angle_of_n_phenomenon_on_left
    }
    new_df = pd.DataFrame.from_dict(new_data)
    new_df = new_df[column_titles]
    if len(old_data) != 0:
      old_df = pd.DataFrame.from_dict(old_data)
      old_df = old_df[column_titles]
      old_df = pd.concat([old_df, new_df], join='inner')
    else:
      old_df = new_df
    old_df.to_csv(filename, index=False)

  def get_rows(start_width, stop_width):
    list_of_widths = list(range(start_width, stop_width))
    rows = df.loc[df[width_key].isin(list_of_widths)]
    return rows

  rows = get_rows(start_width, stop_width)
  if rows.empty or (len(rows) < stop_width - start_width):
    # this is slow, check slice first
    for width in range(start_width, stop_width):
      result = df.loc[df[width_key]==width]

      sample_times.append(dt.datetime.now())

      if not result.empty:
        result = result.values
        time_since_last_computation = (sample_times[1] - sample_times[0]).total_seconds()
      else:
        search_automata = OneDCellularAutomataWithAngleDiscoveryAtMiddle(
          machine_cls=Rule30,
          wall_cls=WallLeftWhiteRightWhite,
          cells_per_generation=width
        )
        discovery = AngleAndDepthDiscovery(search_automata)
        time_since_last_computation = (sample_times[1] - sample_times[0]).total_seconds()

        print("width: {}, angle: {}, depth: {}, tlc_sec: {}".format(width, 
          discovery.n_angle,
          discovery.queue_depth,
          time_since_last_computation))

        width_of_automata.append(width)
        queue_depth.append(discovery.queue_depth)
        angle_of_n_phenomenon_on_left.append(discovery.n_angle)

        if time_since_last_computation > 60:
           update_csv_file(old_data, width_of_automata, queue_depth, angle_of_n_phenomenon_on_left)

    update_csv_file(old_data, width_of_automata, queue_depth, angle_of_n_phenomenon_on_left)
    df = pd.read_csv(filename)
    rows = get_rows(start_width, stop_width)

  # ListLike
  result = ListLike(start_width, stop_width, rows.to_dict('record'))
  return result
    
class OneDCellularAutonomataWallRecursion(OneDCellularAutomata):

  def __init__(self, 
      generations, 
      cells_per_generation=None, 
      initial_condition_index=None,
      machine_cls=None,
      wall_cls=None,
      queue_depth=None):
    '''Translate the information from the center of an automata onto the walls
       via a ring buffer tracking its center information.

    **Args**:
       | ``generations`` (int):  the number of generations over which to run the automata
       | ``cell_per_generation`` (int): how wide the automata is (walls on either side)
       | ``initial_condition_index`` (int): where to place the first black square
       | ``machine_cls`` (Rule): what type of rule machine to place in the bulk
       | ``wall_cls`` (Wall): what type of wall to put on the sides

    **Returns**:
       (type): 

    **Example(s)**:

    .. code-block:: python

      machine = OneDCellularAutonomataWallRecursion(
                  machine_cls=Rule30,
                  wall_cls=WallLeftWhiteRightWhite,
                  queue_depth=6,
                  generations=1000,
                  cells_per_generation=22)

      canvas = Canvas(machine)
      canvas.run_animation(1000, interval=100)
      canvas.save("file_name.pdf")

    '''
    super().__init__(
      generations,
      cells_per_generation,
      initial_condition_index,
      machine_cls,
      wall_cls)

    half_point = round(self.cells_per_generation/2.0)
    self.core_machine_index = half_point

    if queue_depth is None:
      queue_depth = 1
    else:
      queue_depth = queue_depth
    self.queue_depth = queue_depth

    self.core_colors = deque(maxlen=queue_depth)
    self.core_code = []
    self.middle_numbers = []
    self.for_pattern_search = [[] for i in range(self.cells_per_generation)]

    for i in range(4):
      self.core_colors.append('white')
      self.core_code.append(0)
    self.wall_cls = WallLeftWhiteRightWhite

  def initial_state(self):
    super().initial_state()
    self.update_core_code()
    self.core_code = [1 if i == 'black' else 0 for i in self.core_colors]
    self.set_wall_class()

  def next_generation(self):
    super().next_generation()
    self.update_core_code()
    row_number = self.generation+1
    for col_number in range(self.Z.shape[1]):
      cell_color = self.Z[row_number, col_number]
      self.for_pattern_search[col_number].append(1.0 if abs(cell_color-Black)<0.01 else 0.0)
    self.set_wall_class()

  def update_core_code(self):
    self.core_colors.append(self.machines[self.core_machine_index].color)
    self.core_code = [1 if i == 'black' else 0 for i in self.core_colors]

  def set_wall_class(self):

    number = 0
    for index, value in enumerate(self.core_code[0:4]):
      number += value * 2**index

    if number == 1:
      cls = WallLeftWhiteRightBlack
    elif number == 2:
      cls = WallLeftBlackRightWhite
    elif number == 3:
      cls = WallLeftBlackRightBlack
    else:
      cls = WallLeftWhiteRightWhite

    self.wall_cls = cls

  def make_and_start_left_wall_machine(self):
    cls = self.wall_cls
    wall = cls()
    wall.start_at(cls.left_wall)
    return wall

  def make_and_start_right_wall_machine(self):
    cls = self.wall_cls
    wall = cls()
    wall.start_at(cls.right_wall)
    return wall

class ODCAWRPeriodDetection(OneDCellularAutonomataWallRecursion):

  def __init__(self, 
      generations, 
      cells_per_generation=None,
      initial_condition_index=None, 
      machine_cls=None,
      wall_cls=None,
      queue_depth=None):
    '''An automata which will raise one of two exceptions if the period is
    found while it is being run in a different iterator.  If the n-phenomenon
    over-runs the bulk an EOFError exception is raised.  If the autocorrelation
    routine finds the period, a RuntimeError exception is raised (to mimic how
    modern generators work).

    **Note**:
       This object is intended to be used by another object to provide the loop
       call to it's coroutine. (next)

    **Args**:
       | ``generations`` (int):  only needed for super
       | ``cells_per_generation=None`` (int): the width of the automata
       | ``initial_condition_index=None`` (int): where to place the first black
       |                                         cell in the first row of the automata
       | ``machine_cls=None`` (int): the 1-D cellular automata in the bulk
       | ``wall_cls=None`` (Wall): the type of wall class
       | ``queue_depth=None`` (int): the queue depth for the recursive walls

    **Returns**:
       (ODCAWRPeriodDetection):  this object

    **Example(s)**:
      
    .. code-block:: python

        search_automata = ODCAWRPeriodDetection(
          generations=2000000,  # end at some really high number so not to run forever
          machine_cls=Rule30,
          wall_cls=WallLeftWhiteRightWhite,
          cells_per_generation=width,
          queue_depth = depth,
        )
        # The PeriodicityDiscovery consumes the ODCAWRPeriodDetection, by
        # running a loop until exceptions are detected, then it converts these
        # exceptions and ODCAWRPeriodDetection data into useful attributes
        # contained within the discovery object.
        discovery = PeriodicityDiscovery(search_automata)

    '''
    
    super().__init__(
       generations,
       cells_per_generation,
       initial_condition_index,
       machine_cls,
       wall_cls,
       queue_depth=queue_depth)

    self.period = None
    self.repeats = None
    self.queue_depth = queue_depth
    self.last_pattern_without_n = None

    self.black_mask = np.array([True], dtype=np.bool)
    self.white_mask = np.array([False], dtype=np.bool)

    # place left wall and next mask object based on wall type
    if self.wall_cls == WallLeftWhiteRightBlack or \
      self.wall_cls == WallLeftWhiteRightWhite:
      self.n_mask = np.concatenate(
         (self.white_mask, self.black_mask), axis=0)
    else:
      self.n_mask = np.concatenate(
         (self.black_mask, self.white_mask), axis=0)

    [self.build_next_mask() for cell in range(self.cells_per_generation)]

  def build_next_mask(self):

    # alternate color
    if self.n_mask[-1] == self.white_mask:
      self.n_mask = np.concatenate(
        (self.n_mask, self.black_mask), axis=0)
    else:
      self.n_mask = np.concatenate(
        (self.n_mask, self.white_mask), axis=0)

    # place the far right wall object based on wall type
    if len(self.n_mask) >= self.cells_per_generation:
      self.n_mask = self.n_mask[0:self.cells_per_generation]
      self.n_mask[-1] = self.white_mask if \
        self.wall_cls == WallLeftBlackRightWhite or WallLeftWhiteRightWhite else \
        self.black_mask

  def period_search(self):

    previous_generation = self.generation+1
    row_to_check = self.Z[previous_generation]
    sub_row_to_check = row_to_check[0:len(self.n_mask)]

    # find the last location where the n-phenonenom was not reaching into the
    # bulk of the automata
    if len(self.n_mask) >= 3:
      if np.all(np.equal(self.n_mask[0:3], sub_row_to_check[0:3]), axis=0) == False:
        self.last_pattern_without_n = self.generations - previous_generation

    # search for the n-phenonenom reaching all the way through the bulk of our 
    # automata
    if len(self.n_mask) == self.cells_per_generation:
      if np.all(self.n_mask == sub_row_to_check):
        # we have found the mask, so the build finished its unique
        # pattern in the last generation
        self.period = self.generations - previous_generation - 1
        raise EOFError

    # if there mask fails, which it will in most cases we
    # do deep search for repeating period here
    column_correlations = []
    for column_number in range(self.cells_per_generation):
      column_correlations.append(autocorrelate(self.for_pattern_search[column_number]))

    collective_correlations = column_correlations[0]
    for correlation in column_correlations[1:]:
      collective_correlations = np.multiply(collective_correlations, correlation)

    candidate_periods = []

    for i in range(10):
      max_index = np.argmax(collective_correlations)
      candidate_periods.append(max_index)
      collective_correlations[max_index] = 0

    # check for period every cells_per_generation times
    if (self.generations - self.generation) % self.cells_per_generation == 0:

      # try all 10 candidates, we might be able to speed up the algorithm
      # if our actual period is being drown out by the noise at the beginning
      # of our list
      for candidate in candidate_periods:

        if candidate == 0:
          break

        # Z is upside down so we have to really think about our indices here
        location = self.generation + 1  # get the lastest location with information
        checking_location = location + candidate  # look back in time

        # if we aren't at the beginning of our run and the row of of lastest
        # information matches with the row canditate cells into the past, then we
        # have found our period
        if checking_location < self.generations:
          period_found = np.all(self.Z[location, :] == self.Z[checking_location, :])
        else:
          period_found = False
     
        if period_found:
          self.period = candidate
          raise RuntimeError

  def next_generation(self):
    super().next_generation()
    self.period_search()

class PeriodicityDiscovery():
  def __init__(self, automata):
    self.generation = automata.make_generation_coroutine()
    while(True):
      try:
        next(self.generation)
      except EOFError:
        self.repeats = False
        break
      except RuntimeError:
        self.repeats = True
        break
    self.period = automata.period
    self.queue_depth = automata.queue_depth
    self.last_pattern_without_n = automata.last_pattern_without_n

def automata_periodicity(width, start_depth, stop_depth=None):
  '''Get the period and information about if an automata repeats, given a width
  and range of the queue_depths described by the input arguments.

  The period [cells] is a measure of the rule 30 bulk entropy unique duration as it is
  being run forever.  If the period repeats, this unique pattern will occur over
  and over again, if this period does not repeat the bulk of the entropy
  produced will be consumed by the n-phenomenon, despite our efforts to hold it
  off with the recursive wall strategy.
  See https://aleph2c.github.io/miros-random/cellular_automata.html for pictures
  details.

  **Note**:
     This function is memoized (cached) in the file named:
     'rule_30_with_white_walls_depth_and_periodicity.txt'.  This file is opened and
     updated with the panda python library.  Every time a novel query is made,
     subsequent calls to the same program will speed up at the cost of a larger
     cache file.

     If a run of of an individual period discovery calculation takes 60 seconds or
     more, the cached file is saved immediately after the calculation is
     completed.  This way the user can stop midway through a long run and not
     loose the data that has been calculated up until that moment of
     cancellation.

  **Args**:
     | ``width`` (int): which width to start our query (included in result)
     | ``start_depth`` (int): which depth to start the query.
     | ``stop_depth=None`` (int): which depth to stop query (not included in result)

  **Returns**:
     (ListLike): a list like object that can be indexes with a depth int to
     return diction information about that depth, for the given width.  The keys
     of the dictionaries are the same as the column names of the
     'rule_30_with_white_walls_depth_and_period.txt' cached file.

  **Example(s)**:
    
  .. code-block:: python
     
     results = aggregate_period(width=22, start_depth=6, stop_depth=9)

     print(results[5])  #=> IndexError exception

     print(results)  # =>
     [{'repeat?': False, 'queue_depth': 6, 'period': 195, 'width_of_automata': 22},
      {'repeat?': False, 'queue_depth': 7, 'period': 133, 'width_of_automata': 22},
      {'repeat?': False, 'queue_depth': 8, 'period': 167, 'width_of_automata': 22}]

     print(results[6])  # =>
     {'repeat?': False, 'queue_depth': 6, 'period': 195, 'width_of_automata': 22}

     print(results[7])  # => ...
     print(results[8])  # => ...

     print(results[9])  #=> IndexError exception

  '''
  if stop_depth is None:
    stop_depth = start_depth + 1

  width_list, depth_list, period_list, repeat_list = [], [], [], []

  sample_times = deque(maxlen=2)
  sample_times.append(dt.datetime.now())
  sample_times.append(dt.datetime.now())

  column_titles = [
    width_key,
    depth_key,
    period_key,
    repeat_key]
  filename = 'rule_30_with_white_walls_depth_and_periodicity.txt'

  dtype0 = {
    width_key: 'int64',
    depth_key: 'int64',
    period_key: 'int64',
    repeat_key: 'bool'
  }

  try:
    df = pd.read_csv(filename)
    df = df.astype(dtype0)
    old_data = df.to_dict()
  except:
    old_data = {}
    df = pd.DataFrame({
      width_key: [],
      depth_key: [],
      period_key: [],
      repeat_key: [],
      })
    df = df[column_titles]
    df = df.astype(dtype0)

  def update_csv_file(old_data, width_of_automata, queue_depth, period, repeats):
    new_data = {
      width_key: width_of_automata,
      depth_key: queue_depth,
      period_key: period,
      repeat_key: repeats,
    }
    new_df = pd.DataFrame.from_dict(new_data)
    new_df = new_df[column_titles]

    if len(old_data) != 0:
      old_df = pd.DataFrame.from_dict(old_data)
      old_df = old_df[column_titles]
      old_df = pd.concat([old_df, new_df], join='inner')
    else:
      old_df = new_df
    old_df.to_csv(filename, index=False)

  def get_rows(width, start_depth, stop_depth):
    list_of_depths = list(range(start_depth, stop_depth))
    dff = df[df[width_key] == width]
    mask = dff[depth_key].isin(list_of_depths)
    rows = dff[mask]
    return rows

  rows = get_rows(width, start_depth, stop_depth)
  if rows.empty or (len(rows) < stop_depth-start_depth):
    for depth in range(start_depth, stop_depth):
      dff = df[df[width_key] == width]
      mask = dff[depth_key].isin([depth])
      result = dff[mask]

      sample_times.append(dt.datetime.now())

      if not result.empty:
        result = result.values
        time_since_last_computation = (sample_times[1] - sample_times[0]).total_seconds()
      else:
        search_automata = ODCAWRPeriodDetection(
          generations=2000000,
          machine_cls=Rule30,
          wall_cls=WallLeftWhiteRightWhite,
          cells_per_generation=width,
          queue_depth = depth,
        )
        discovery = PeriodicityDiscovery(search_automata)

        time_since_last_computation = (sample_times[1] - sample_times[0]).total_seconds()
        print("width: {}, last_bulk: {}, period: {}, depth: {}, tlc_sec: {}".format(
          width, 
          discovery.last_pattern_without_n,
          discovery.period,
          discovery.queue_depth,
          time_since_last_computation))

        width_list.append(width)
        depth_list.append(depth)
        period_list.append(discovery.period)
        repeat_list.append(discovery.repeats)

        if time_since_last_computation > 60:
           update_csv_file(old_data, 
             width_list, 
             depth_list,
             period_list,
             repeat_list)

    update_csv_file(old_data, width_list, depth_list, period_list, repeat_list)
    df = pd.read_csv(filename)
    rows = get_rows(width, start_depth, stop_depth)

  # ListLike
  result = ListLike(start_depth, stop_depth, rows.to_dict('record'))
  return result


class Canvas():
  def __init__(self, automata, title=None):
    '''Animate 2D graphing paper, or static file describing a automata

    Given an ma, which has a ``make_generation_coroutine`` coroutine generator, an
    animation can be build by calling this coroutine for as many generations are
    required.

    **Note**:
       This ``automata`` object needs to provide a ``make_generation_coroutine`` method which
       returns a coroutine which can be called with ``next``.

    **Args**:
       | ``automata`` (OneDCellularAutomata): 
       | ``title=None`` (string): An optional title

    **Returns**:
       (Canvas): this object

    **Example(s)**:
      
    .. code-block:: python
       
       eco1 = Canvas(ma)
       eco1.run_animation(1200, interval=10)  # 10 ms
       eco1.save('eco1.mp4')

       eco2 = Canvas(automata)
       eco2 = save('eco2.pdf, generations=100)

    '''
    self.fig, self.ax = plt.subplots()
    if title:
      self.ax.set_title(title)
    self.automata = automata
    self.generation = automata.make_generation_coroutine()
    self.ax.set_yticklabels([])
    self.ax.set_xticklabels([])
    self.ax.set_aspect(1.0)
    self.ax.xaxis.set_ticks_position('none')
    self.ax.yaxis.set_ticks_position('none')
    self.fig.tight_layout()
    # seventies orange/browns looking color map
    self.cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
      'oranges', ['#ffffff', '#ffa501', '#b27300', '#191000'])
    self.grid = self.ax.pcolormesh(next(self.generation), cmap=self.cmap)

  def init(self):
    '''animation initialization callback

    **Note**:
       This not needed by our animation, but it is needed by the library we are
       calling, so we just stub it out

    **Returns**:
       (tuple): (self.grid,)

    '''
    return (self.grid,)

  def animate(self, i):
    '''animation callback.

    This method will be called for each i frame of the animation.  It creates
    the next generation of the automata then it updates the pcolormesh using the
    set_array method.

    **Args**:
       | ``i`` (int): animation frame number

    **Returns**:
       (tuple): (self.grid,)

    '''
    self.Z = next(self.generation)
    # set_array only accepts a 1D argument
    # so flatten Z before feeding it into the grid arg
    self.grid.set_array(self.Z.ravel())
    return (self.grid,)
  
  def run_animation(self, generations, interval):
    '''Run an animation of the automata.

    **Args**:
       | ``generations`` (int): number of automata generations
       | ``interval`` (int): movie frame interval in ms

    **Example(s)**:

    .. code-block:: python

      eco = Canvas(automata)
      eco.run_animation(1200, interval=20)  # 20 ms

    '''
    self.anim = animation.FuncAnimation(
      self.fig, self.animate, init_func=self.init,
      frames=generations, interval=interval,
      blit=False)

  def save(self, filename=None, generations=0):
    '''save an animation or run for a given number of generations and save as a
       static file (pdf, svg, .. etc)

    **Note**:
       This function will save as many different static file formats as are
       supported by matplot lib, since it uses matplotlib.

    **Args**:
       | ``filename=None`` (string): name of the file
       | ``generations=0`` (int): generations to run if the files doesn't have a
       |                          'mp4' extension and hasn't been animated before


    **Example(s)**:

       eco1 = Canvas(ma)
       eco1.run_animation(50, 10)
       eco1.save('rule_30.mp4)
       eco1.save('rule_30.pdf)

       eco2 = Canvas(ma)
       eco1.save('rule_30.pdf', generations=40)

    '''
  def save(self, filename=None, generations=0, dpi=100):

    if pathlib.Path(filename).suffix == '.mp4':
      self.anim.save(filename) 
    else:
      if self.automata.generation > 0:
        for i in range(self.automata.generations):
          try:
            next(self.generation)
          except RuntimeError:
            break
        self.ax.pcolormesh(self.automata.Z, cmap=self.cmap)
      #for item in [self.fig, self.ax]:
      #  item.patch.set_visible(False)
      self.ax.axis('off')
      plt.savefig(filename, dpi=dpi) 

  def close(self):
    plt.close(self.fig)

class ODCAXEquation:
  def __init__(self, odcavar, generations, width_alg='max'):
    self.automata = [odcavar]
    self.width_alg = width_alg
    self.cells_per_generation = odcavar.cells_per_generation
    self.max_width = odcavar.cells_per_generation
    self.min_width = odcavar.cells_per_generation
    self.generations = generations

  def load(self, odcarvar):
    self.automata.append(odcarvar)
    # the max_width is the maximum max_width
    self.max_width = max([automata.cells_per_generation for automata in self.automata])
    self.min_width = min([automata.cells_per_generation for automata in self.automata])
    self.cells_per_generation = self.max_width if self.width_alg == 'max' else self.min_width
    self.cells_per_generation = self.cells_per_generation

  def __xor__(self, vector):
    self.load(vector)
    return self

  def pad_width_to_max(self, vector):
    '''Wrap pad a row in the provided vector

    **Note**:
       Do this not that recommendation

    **Args**:
       | ``vector`` (1d ndarray): The vector row you want to pad

    **Returns**:
       (1d ndarray): The vector padded so that its width is self.width.  The
       begging part of the vector is placed on the end until the row width of
       the vector is equal to the self.width of this obj
    '''
    padding_needed = self.max_width - vector.size
    if padding_needed > 0:
      # keep all the rows and get rid of the first columns added by the pad command
      result = np.lib.pad(vector, (0, padding_needed), mode='wrap')
    else:
      result = vector
    return result

  def trim_width_to_min(self, vector):
    size_needed = vector.size - (vector.size - self.min_width)
    result = vector[0:size_needed]
    return result

  def initial_state(self):
    '''initialize the 2d cellular automata'''

    if self.width_alg == 'max':
      self.for_pattern_search = [[] for i in range(self.max_width)]
      self.Z = np.full([self.generations, self.max_width], Black, dtype=np.bool)
    else:
      self.for_pattern_search = [[] for i in range(self.min_width)]
      self.Z = np.full([self.generations, self.min_width], Black, dtype=np.bool)

    [obj.initial_state() for obj in self.automata]
    # pull the last row vector out of each automata and make a list of vectors
    # multiply each of these vectors together for the row_result
    g = self.automata[0].generation
    self.row_result = reduce(self.xor_row, [obj.Z[g, :] for obj in self.automata])
    self.Z[0, :] = self.row_result
    self.generation = self.automata[0].generation

  def xor_row(self, row1, row2):
    if self.width_alg == 'max':
      padded_row1 = self.pad_width_to_max(row1)
      padded_row2 = self.pad_width_to_max(row2)
    else:
      padded_row1 = self.trim_width_to_min(row1)
      padded_row2 = self.trim_width_to_min(row2)
    return np.bitwise_xor(padded_row1, padded_row2)

  def make_generation_coroutine(self):

    self.initial_state()
    yield self.Z
    while True:
      g = self.automata[0].generation  # since the other iterator incremented it
      [obj.next_generation() for obj in self.automata]
      # pull the last row vector out of each automata and make a list of vectors
      # multiply each of these vectors together for the row_result
      if g != self.generations:
        self.Z[g, :] = reduce(self.xor_row, [obj.Z[g, :] for obj in self.automata])
        self.generation = self.automata[0].generation
        for col_number in range(self.Z.shape[1]):
          cell_color = self.Z[g, col_number]
          self.for_pattern_search[col_number].append(1.0 if cell_color else 0.0)
      yield self.Z

if __name__ == '__main__':

  generations = 1000

  # the 210 periodicity of a b c snuck through in the a b c d analysis
  a = OneDCellularAutonomataWallRecursion(cells_per_generation=10, queue_depth=4, generations=generations)
  b = OneDCellularAutonomataWallRecursion(cells_per_generation=10, queue_depth=7, generations=generations)
  c = OneDCellularAutonomataWallRecursion(cells_per_generation=11, queue_depth=3, generations=generations)
  d = OneDCellularAutonomataWallRecursion(cells_per_generation=12, queue_depth=5, generations=generations)

  # a ^ b ^ c == c ^ b ^ a
  # commutative (order) indifferent
  # associative (grouping) indifferent
  equation = ODCAXEquation(c, generations=generations, width_alg='min')
  equation ^= b
  equation ^= a
  equation ^= d

  print(periodicity(b, c, a, d))

  width = 22
  queue_depth = 2
  ma = OneDCellularAutonomataWallRecursion(
    generations=generations,
    machine_cls=Rule30WithQueueDepth,
    cells_per_generation=width,
    queue_depth=queue_depth,
    )

  filename = "equation_rec_walls_{}_queue_{}_gen_{}".format(width, queue_depth, generations)
  thing_to_render = equation
  eco = Canvas(thing_to_render)
  eco.run_animation(generations, interval=100)
  movie_filename = '{}.mp4'.format(filename)
  eco.save(movie_filename)
  png = '{}.png'.format(filename)
  eco.save(png)

  eco.save('{}.pdf'.format(filename))
  eco.close()
  cmd = 'cmd.exe /C {} &'.format('{}.pdf'.format(filename))
  subprocess.Popen(cmd, shell=True)

  # a specific column can repeat, while the other columns change
  # for this reason we need to multiply the spectrums together so
  # as to find where the real pattern repetitions take place
  max_c_indexs = []
  column_correlations = []
  for i in range(thing_to_render.cells_per_generation):
    column_correlations.append(autocorrelate(thing_to_render.for_pattern_search[i]))
    max_index = np.argmax(column_correlations[-1])
    max_c_indexs.append(max_index)

  collective_correlations = column_correlations[0]
  for correlation in column_correlations[1:]:
    collective_correlations = np.multiply(collective_correlations, correlation)

  fig = plt.figure()
  autocorrelation_filename = "autocorrection.pdf"
  ##plt.plot(pattern_index, collective_autocorrelation_fft_product)
  ##plt.plot(pattern_index, cc)
  plt.plot([i for i, j in enumerate(collective_correlations)], collective_correlations)
  plt.savefig(autocorrelation_filename, dpi=300)

  of_interest = []
  for i in range(10):
    max_index = np.argmax(collective_correlations)
    of_interest.append(max_index)
    collective_correlations[max_index] = 0

  print(of_interest)

  cmd = 'cmd.exe /C {} &'.format(movie_filename)
  subprocess.Popen(cmd, shell=True)

  cmd = 'cmd.exe /C {} &'.format(autocorrelation_filename)
  subprocess.Popen(cmd, shell=True)


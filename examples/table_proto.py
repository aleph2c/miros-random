import os
import sys
from math import factorial
from functools import wraps
from operator import attrgetter
from itertools import combinations
from collections import namedtuple

import pprint
def pp(item):
  pprint.pprint(item)

path_to_this_file = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path_to_this_file + '/../')

from is_prime import is_prime
from random_number import automata_periodicity

TableSpec = namedtuple('TableSpec', [
  'cells_per_generation',
  'queue_depth',
  'period',
  'loop'])

class MiniTable():
  Table_Upper_Size_Limit = 15

  @staticmethod
  def automata_periodicity_to_table_spec(ap_dict):
    '''converts a automata_periodicity result dictionary into a TableSpec
    namedtuple'''
    result = TableSpec(
      cells_per_generation=ap_dict['width_of_automata'],
      queue_depth=ap_dict['queue_depth'],
      period=ap_dict['period'],
      loop=ap_dict['repeat?'])
    return result

  class Dectorators:
    '''decorators used by the MiniTable class'''
    @staticmethod
    def dict_to_spec(fn):
      '''decortator which converts the results of the automata_periodicity results
      into a table spec'''
      @wraps(fn)
      def _dict_to_spec(self, spec):
        # convert a dict object into the TableSpec namedtuple, but only if the
        # argment provided is a dictionary
        if type(spec) == dict:
          spec = MiniTable.automata_periodicity_to_table_spec(spec)
        return fn(self, spec)
      return _dict_to_spec

  def __init__(self):
    '''short description

    longer description

    **Note**:
       Do this not that recommendation

    **Args**:
       | ``x`` (type1): 

    **Returns**:
       (type): 

    **Example(s)**:
      
    .. code-block:: python

      mt = MiniTable()
      [mt.add_item(result) for result in automata_periodicity(width=22, start_depth=3, stop_depth=22).list]
      # the construction of the combinations can take a long time, so we use
      # generators and coroutines, so we can get access to the computation and
      # the memory at the momement we need it
      coroutine = mt.make_combination_coroutine(r=3)
      while(True):
        try:
          result, maximum_scalar, row_scalar, r = next(coroutine)
          print(result)  # output the current spec
          print(maximum_scalar) # how far away from the maximum burst of
                                # combinations we are for the last item added to
                                # the table (0.0 to 1.0)... this is parabolic
                                # function who's maximum value is at 1.
          print(row_scalar) # output of how far we are through the current row
                            # for our current r value
          print(r)  # the current number of picks to be made from the self.table
                    # to construct a specification from
        except RuntimeError:
          break
    '''
    # the collection to make combinations from
    self.table = []
    # an item which must be included in each combination source from this object
    self.root = None

  @Dectorators.dict_to_spec
  def add_item(self, element):
    '''
    Add a table item of TableSpec to the table.  If the table exceeds the
    MiniTabe.Table_Upper_Size_Limit value, remove the lowest value item before
    appending the new item.

    **Note**:
       This function will not evaluate the worth of the new item agains the
       existing items, it assumes that it is important and it will not be
       dropped from the table if there isn't enough room (until the next call to
       this method)

    **Args**:
       | ``element`` (TableSpec): A machine specification, with characterists
       which is to be added to the table.

    **Example(s)**:
      
    .. code-block:: python

      mt.MiniTable()
      mt.add_item(TableSpec(
        cells_per_generation=10,
        queue_depth=2,
        period=27,
        loop=True)

    '''
    for field in TableSpec._fields:
      if not hasattr(element, field):
        raise AttributeError

    # don't return something that doesn't loop
    if not element.loop:
      return

    self.count = len(self.table)
    if self.count == MiniTable.Table_Upper_Size_Limit:
      self._remove_item()

    self.table.append(element)
    self.root = element

  def _remove_item(self):
    '''Remove a table item which has the lowest non-prime period

    **Note**:
       If a table entry doesn't loop, it is removed first
    '''
    removal_candidate = None
    sorted_table = sorted(self.table, key=attrgetter('loop'))
    for table_item in sorted_table:
      if not table_item.loop:
        removal_candidate = table_item
        break
      else:
        break

    if removal_candidate == None:
      sorted_table = sorted(sorted_table, key=attrgetter('period'))
      for table_item in sorted_table:
        if not is_prime(table_item.period):
          removal_candidate = table_item
          break
    self.table.remove(removal_candidate)

  def make_combinations_coroutine(self):
    '''Make a coroutine which will return a new specification when it is used as
    an argument to the next keyword.

    For each call to this coroutine a combination will be returned.  The
    combinations will be constructed this table, where n = len(table) and for r
    picks of items, where r will range from 2 to n, and the last item added to
    the table, via the add_item method, is guaranteed to will be included in
    every combination.

    **Note**:
       If the table size is zero a RuntimeError exception is raised

       When the number of possible combinations is exhausted the coroutine will
       issue a RuntimeError exception.

    **Returns**:
      | (coroutine):  a coroutine which must be used as an argument to the
      |               ```next`` Python keyword.  The results of the next
      |               operation will return a TableSpec. (see example)

    **Example(s)**:
      
    .. code-block:: python

      mt = MiniTable()

      [mt.add_item(result) for 
        result in 
        automata_periodicity(
          width=22, start_depth=3, stop_depth=22).list
      ]

      coroutine = mt.make_combinations_coroutine()

      while(True):
        try:
          result, maximum_scalar, row_scalar, r = next(coroutine)
          print(result)  # output the current spec
          print(maximum_scalar) # how far away from the maximum burst of
                                # combinations we are for the last item added to
                                # the table (0.0 to 1.0)... this is parabolic
                                # function who's maximum value is at 1.
          print(row_scalar) # output of how far we are through the current row
                            # for our current r value
          print(r)  # the current number of picks to be made from the self.table
                    # to construct a specification from
        except RuntimeError:
          break

    '''
    r = 2
    max_r = self.picks_needed_to_get_maximum_combinations()
    coroutine = self.make_combination_coroutine(r)
    while(True):

      if r > self.count:
        raise RuntimeError

      # maximum_scalar will change parabolically, from zero to 1 back to zero.
      # It's high point will be achieved when we are at an r which maximizes the
      # the available combinations from our table.
      maximum_scalar = -1.0*(1/float(max_r)*r-1)**2 + 1.0
      try:
        # the row_scalar varies from 0 to 1 depending on how far into the
        # current row iteration we are
        result, row_scalar = next(coroutine)

        yield(result, maximum_scalar, row_scalar, r) 
      except RuntimeError:
        r += 1
        coroutine = self.make_combination_coroutine(r)


  def make_combination_coroutine(self, r):
    '''make a coroutine which will return a new specification when it is added
    to the next keyword as an argument.

    **Note**:
       If the table size is zero a RuntimeError exception is raised

       When the number of possible combinations is exhausted the coroutine will
       issue a RuntimeError exception.

    **Args**:
       | ``r`` (int): number of r items to be picked from the total selection at a time.

    **Returns**:
       | (coroutine): an item which must be used as an argument to the ``next``
       |              Python keyword

    **Example(s)**:
      
    .. code-block:: python
       
      mt = MiniTable()
      [mt.add_item(result) for result in automata_periodicity(width=22, start_depth=3, stop_depth=22).list]
      coroutine = mt.make_combination_coroutine(r=3)
      while(True):
        try:
          result, row_scalar = next(coroutine)
          print(result)  # output the spec
          print(row_scalar)  # output how far through this combination we are 0 to 1
        except RuntimeError:
          break

    '''
    
    total = float(self.num_of_coms_for_pick(r))
    r = r - 1
    if len(self.table) == 0:
      raise RuntimeError
    counter = 0
    if self.root == None:
      raise RuntimeError
    sub_table = self.table[:]
    sub_table.remove(self.root)
    _combination_of_table = combinations(sub_table, r)

    for item in _combination_of_table:
      counter += 1
      yield (item + (self.root,), counter/total)
    raise RuntimeError

  def num_of_coms_for_pick(self, r):
    '''the number of combinations for n items in the table, (where the self.root item is
    included in each selection), taken r at a time'''
    n = len(self.table) - 1
    r = r - 1
    num = factorial(n)
    den = factorial(n-r)
    den *= factorial(r)
    if den:
      ans = num/den
    else:
      ans = 0
    return ans

  def picks_needed_to_get_maximum_combinations(self):
    '''Given the current table, what number r (items to be picked from the total
    available table items at a time) will produce the largest number of
    combination.  This assumes that the self.root value (the value of the last
    argument given to the add_item method) will be included in each combination
    source from the coroutine building the combinations.
    
    **Note**:
      The result of this equation was determined by differentiating the combinations
      equations for this class, then setting it's results to zero.
    
    '''
    # print(round((len(self.table)+0.000001)/2.0))
    # add a bit to make our round function work the way we want with floats
    return round((len(self.table)+0.000001)/2.0)
 
if __name__ == "__main__":
  mt = MiniTable()
  results = [MiniTable.automata_periodicity_to_table_spec(item) for item in automata_periodicity(width=22, start_depth=3, stop_depth=22).list]
  #coroutine = None
  [mt.add_item(result) for result in automata_periodicity(width=22, start_depth=3, stop_depth=22).list]
  #[mt.add_item(result) for result in results]
  coroutine = mt.make_combinations_coroutine()
  #coroutine = mt.make_combination_coroutine(r=2)
  #print(mt.num_of_coms_for_pick(3))
  #print(mt.picks_needed_to_get_maximum_combinations())

  while(True):
    try:
      result, maximum_scalar, row_scalar, r = next(coroutine)
      print(result)
      print(maximum_scalar)
      print(row_scalar)
      print(r)
    except RuntimeError:
      break
  
  a = 1
   


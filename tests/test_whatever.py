import os
import sys
import pytest

# pytest is too stupid to find files
path_to_this_file = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path_to_this_file + '/../')

from whatever import Whatever

@pytest.mark.whatever
def test_whatever():
  w = Whatever()
  result = w.whatever()
  assert(result == "something")

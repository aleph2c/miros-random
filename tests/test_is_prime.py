import os
import sys
import pytest

# pytest is too stupid to find files
path_to_this_file = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, path_to_this_file + '/../')

from is_prime import is_prime

@pytest.mark.is_prime
def test_simple_prime():
  assert(is_prime(53) == True)
  assert(is_prime(57) == False)

@pytest.mark.is_prime
def test_last_prime():
  assert(is_prime(15485863) == True)

@pytest.mark.is_prime
def test_last_non_prime():
  assert(is_prime(15485863-1) == False)
  assert(is_prime(15485863+1) == False)


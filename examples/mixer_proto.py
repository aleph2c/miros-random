import time
import random
from collections import deque
from itertools import combinations

from miros import Event
from miros import Factory
from miros import signals
from miros import return_status

# Goal build an active object that has a synchronous methods
# which gets its results from asynchronous processes.  (We need to share data
# between two threads)
class Mixer(Factory):

  Name = "mini_mixer"
  Heart_Beat_Sec = 0.50
  Default_Buffer_Size = 5

  def __init__(self, name=None, buffer_size=None, heart_beat_sec=None, live_trace=None, live_spy=None):

    super().__init__(name if name != None else Mixer.Name)
    self.live_trace = live_trace if live_trace != None else False
    self.live_spy = live_spy if live_spy != None else False

    self.buffer_size = buffer_size if \
      buffer_size != None else \
      Mixer.Default_Buffer_Size

    self.heart_beat_sec = heart_beat_sec if \
      heart_beat_sec != None else \
      Mixer.Heart_Beat_Sec

    self.buffer = deque(maxlen=self.buffer_size)

    self.common_behavior = self.create(state="common_behavior"). \
      catch(signal=signals.ENTRY_SIGNAL,
        handler=self.common_features_entry). \
      catch(signal=signals.pioneer,
        handler=self.common_features_pioneer). \
      to_method()

    self.pioneer_new_spec = self.create(state="pioneer_new_spec"). \
      catch(signal=signals.ENTRY_SIGNAL,
        handler=self.pioneer_new_entry). \
      catch(signal=signals.done,
        handler=self.pioneer_new_spec_done). \
      to_method()

    self.nest(self.common_behavior, parent=None). \
      nest(self.pioneer_new_spec, parent=self.common_behavior)

    self.start_at(self.common_behavior)
    time.sleep(Mixer.Heart_Beat_Sec)

  @staticmethod
  def common_features_entry(mm, e):
    status = return_status.UNHANDLED
    for i in range(mm.buffer_size):
      mm.post_fifo(Event(signal=signals.pioneer))
    return status

  @staticmethod
  def common_features_pioneer(mm, e):
    status = mm.trans(mm.pioneer_new_spec)
    return status

  @staticmethod
  def pioneer_new_entry(mm, e):
    status = return_status.UNHANDLED

    # create an asynchronous process that fills the same buffer that our synchronous
    # wants to read (spec_pioneer_time is in a resolution of 10 ms)
    spec_pioneer_time = random.randint(1, Mixer.Heart_Beat_Sec*100.0)/100.0
    time.sleep(spec_pioneer_time)

    # just use the spec pioneer time to fill the data our outside process wants
    # to look at
    mm.buffer.append(spec_pioneer_time)
    self.post_lifo(Event(signal=signals.done))
    return status

  @staticmethod
  def pioneer_new_spec_done(mm, e):
    status = mm.trans(mm.common_behavior)
    return status
    
  def get_spec(self):
    if len(self.buffer) == 0:
      raise LookupError
    spec = self.buffer.popleft()
    return spec

  def get_spec_blocking(self):
    # if there is no item to give them, block for our 
    # Heart_Beat_Sec and then try again, if there is 
    # still nothing, raise an exception
    if len(self.buffer) == 0:
      time.sleep(Mixer.Heart_Beat_Sec)
      if len(self.buffer) == 0:
        raise LookupError
    spec = self.buffer.popleft()
    return spec
    
if __name__ == "__main__":
  mm = Mixer(buffer_size=10, live_trace=True)

  for i in range(mm.buffer_size):
    time.sleep(Mixer.Heart_Beat_Sec)
    print(mm.get_spec())

  mm = Mixer(buffer_size=10, live_trace=True)
  for i in range(mm.buffer_size):
    print(mm.get_spec_blocking())

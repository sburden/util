
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public
#  License as published by the Free Software Foundation; either
#  version 3.0 of the License, or (at your option) any later version.
#
#  The library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  General Public License for more details.
#
# (c) Sam Burden, UC Berkeley, 2013 

import os
from time import time
from glob import glob

def file(fi,di='',sfx=''):
  """
  Find previous datestring file for given file prefix

  Inputs
    fi - str - file name prefix
    (optional)
    di - str - directory containing previous datestring data
    sfx - str - previous datestring filetype suffix

  Outputs
    fi - str - previous datestring file corresponding to pfx
  """
  fis = glob( os.path.join(di, '*'+sfx) )
  if '_' in fi:
    fi,_ = fi.split('_')
  S = lambda s : os.path.split( s )[1].strip(sfx) # strip filename
  I = lambda s : int( s.replace('-','') ) # convert datestring to int
  # compute integer datestrings
  i = I(fi)
  F = [S(f) for f in fis if I( S( f ) ) <= i]
  assert F # fails if no files found
  return os.path.join( di, F[-1]+sfx )

def latest(di,sfx='',sec=False):
  """
  Find latest datestring file in a directory

  Inputs
    di - str - directory 
    sfx - str - filetype suffix
    (optional)
    sec - bool - whether to include sec in output

  Outputs
    fi - str - latest file
  """
  return file(datestring(sec=sec),di=di,sfx=sfx)

def datestring(t=None,sec=False):
  """
  Datestring

  Inputs:
    (optional)
    t - time.localtime()
    sec - bool - whether to include sec [SS] in output

  Outputs:
    ds - str - date in YYYYMMDD-HHMM[SS] format

  by Sam Burden 2012
  """
  if t is None:
    import time
    t = time.localtime()

  ye = '%04d'%t.tm_year
  mo = '%02d'%t.tm_mon
  da = '%02d'%t.tm_mday
  ho = '%02d'%t.tm_hour
  mi = '%02d'%t.tm_min
  se = '%02d'%t.tm_sec
  if not sec:
    se = ''

  return ye+mo+da+'-'+ho+mi+se

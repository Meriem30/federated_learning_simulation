import os
import sys

# return the directory name (of the canonical path of the module resolving any symbolic links)
currentdir = os.path.dirname(os.path.realpath(__file__))  # hold the absolute path to the __init__ directory

# construct and insert (at the beginning) the path to the parent directory of the currentdir
sys.path.insert(0, os.path.join(currentdir, ".."))

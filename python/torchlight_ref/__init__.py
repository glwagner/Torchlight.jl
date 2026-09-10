"""Torchlight reference implementations and fixture exporters (Python side).

The Python side owns the *source* framework behaviour: model definitions,
forward/derivative/training evaluation, and HDF5 fixture export.  Nothing in
the Julia package imports Python at runtime; fixtures are the only interface.
"""

SCHEMA_VERSION = 1

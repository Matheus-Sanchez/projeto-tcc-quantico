"""Local quantum vector heads.

Keep this module free of TensorFlow imports: simulation workers use spawn and
must never initialize a second TensorFlow runtime.
"""

"""
auto_actuary
============
Actuarial analytics platform for P&C carriers.

FCAS-level mathematics. Executive-grade output.

Quick start
-----------
>>> from auto_actuary import ActuarySession
>>> session = ActuarySession.from_config("config/schema.yaml")
>>> session.load_csv("policies", "data/policies.csv")
>>> session.load_csv("valuations", "data/valuations.csv")
>>> tri = session.build_triangle(lob="PPA", value="incurred_loss")
>>> tri.develop()
>>> print(tri.ultimates())
"""

from auto_actuary.core.session import ActuarySession  # noqa: F401

__version__ = "0.1.0"
__all__ = ["ActuarySession"]

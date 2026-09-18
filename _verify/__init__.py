"""Verification helpers and the strict benchmark dispatch module.

The public ``yonod.py`` entry point imports ``_verify.run_benchmark`` for the
serial manifest-outer-CV runtime.  Keeping this directory an explicit package
ensures the dispatch is stable on every supported Python version.
"""

#!/bin/bash

# Unit tests
pytest -v -s -x seaice_ecdr/tests/unit/test_initial_daily_ecdr.py

# Integration tests
pytest -v -s -x seaice_ecdr/tests/integration/test_initial_daily_ecdr_generation.py

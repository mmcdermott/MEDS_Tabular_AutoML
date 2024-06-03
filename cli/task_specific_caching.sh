#!/usr/bin/env bash

echo "Caching Training Data for Task"
python scripts/task_specific_caching.py "$@"

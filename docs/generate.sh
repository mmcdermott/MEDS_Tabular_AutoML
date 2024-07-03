#!/usr/bin/env bash
# bash generate.sh

set -e

rm -rf build
make html
cd build/html
python -m http.server

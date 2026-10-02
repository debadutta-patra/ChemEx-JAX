#!/bin/sh

chemex fit -e Experiments/*.toml \
           -p Parameters/parameters.toml \
           -d 2st \
           -o Output

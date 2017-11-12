#!/usr/bin/env python
# encoding: utf-8

import reda.importers.bert
import os

path = "data"
files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".ohm")]

files.sort()
print("Found %d files in %s: %s" % (len(files), path, files))

for fname in files:
    reda.importers.bert.load(fname)


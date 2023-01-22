#!/usr/bin/env ruby

# SPDX-License-Identifier: 0BSD

$VERBOSE = true

require 'json'

def run_check
  python_path = File.expand_path('../python/python-embedding.json',  __dir__)
  java_path = File.expand_path('../java/java-embedding.json', __dir__)
  equal = JSON(File.read(python_path)) == JSON(File.read(java_path))
  printf "The embeddings %s equal.\n", (equal ? 'ARE' : 'are NOT')
end

run_check if $PROGRAM_NAME == __FILE__

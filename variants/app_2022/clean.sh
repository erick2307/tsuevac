#!/bin/zsh
rm -r $1
rm -r ./setup/tmp
rm -r ./setup/data
echo "Removed $1 tmp and data directories"

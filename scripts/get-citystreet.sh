#!/usr/bin/env bash
# Download the CityStreet dataset from its official Google Drive folder
# (linked from https://visal.cs.cityu.edu.hk/research/citystreet/).
# Fetches only CityStreet files: the bundled DukeMTMC and PETS2009 archives are skipped.
# Needs uv (https://docs.astral.sh/uv/) and about 32 GB free disk (16 GB archive + extraction).
set -euo pipefail

dest="${1:-data/citystreet}"
mkdir -p "$dest"
gdown() { nice -n 15 uvx -q gdown "$@"; }

gdown 1qQHQ_uoGRuzLst8FJJK8PJj3GQoEZJ1E -O "$dest/COPYRIGHT.txt"
gdown 1xFraMurDVajuBWjzVityu4G3utQUmVLO -O "$dest/CityStreet_Multi-View Crowd Counting Dataset.pdf"
gdown --folder 1Bg4S_OTFKYVEQ6BtARMi55nUo9bQvnUP -O "$dest/CityStreet_corrected_labels"
gdown --folder 1WLarleM422lLSRsfVvC-vwzqmzeRnwib -O "$dest/CityStreet_zip"

echo "Downloaded to $dest. Extract with:"
echo "  cd $dest/CityStreet_zip && zip -s 0 CityStreet.zip --out single.zip && unzip single.zip"

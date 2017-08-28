# Transcription Factor Mining: An Active Learning Approach

This repository contains code, results and supplementary data for my the Master's thesis "Transcription Factor Mining: An Active Learning Approach"

* `src` contains all Python scripts used to pre-process, learn and evaluate
* `preprocessed_training_data` contains the pre-processed training data
* `resources` offers additional data and software used
* `results` has all the measurements as well as iPython notebooks that transform the results into pretty charts
* `assets`: location of final charts, drawings and bibtex file

## Setup Python

* install Python 3 and pip
* run `pip install -r requirements.txt` to install dependencies

## Setup libact

* checkout and install libact from [this](https://github.com/byted/libact) fork

## Building jSRE

* go to `resources/jsre` and run `ant install`

## Setting Up BBLIP Model files

* go to `resources/McClosky-2009` and run `tar xzvf bioparsingmodel-rel1.tar.gz`
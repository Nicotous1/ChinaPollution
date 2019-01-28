# Predicting pollution in China

*Authors: Nicolas Toussaint, Alexis Saïr, Robin Fuchs, Ambroise Coulomb, Enzo Terreau, Antoine Hoorelbeke*

## Introduction

PM2.5 refers to atmospheric particulate matter (PM) that have a diameter of less than 2.5 micrometers, which is about 3% the diameter of a human hair.

![pm25_comparison]

Since they are so small and light, fine particles tend to stay longer in the air than heavier particles. This increases the chances of humans and animals inhaling them into the bodies.

Studies have found a close link between exposure to fine particles and premature death from heart and lung disease. Fine particles are also known to trigger or worsen chronic disease such as asthma, heart attack, bronchitis and other respiratory problems.

Knowing and anticipating the level of PM2.5 in the air is a major public health issue.

## Set up

1. Install the ramp-workflow library (if not already done)  
``
$ pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git
``

2. Follow the ramp-kits instructions from the [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki) 


## EDA 

This [notebook](./Exploratory-Data-Analysis.ipynb) is an exploratory analysis of the data available for this challenge. It's a first step to understand the data

## Starting kit

A first model has been designed in this [folder](./submissions/starting_kit). It is composed of a feature extractor that is performing preprocessing of the data and a regressor that is a Ridge regression.

To test the starting kit:  

``
$ ramp_test_submission --submission starting_kit
``

## Help

Go to the [ramp-workflow wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki) for more help on the RAMP ecosystem.


## References

https://blissair.com/what-is-pm-2-5.html

[pm25_comparison]: ./img/pm25_comparison.jpg

# Predicting pollution in China

*Authors: Nicolas Toussaint, Alexis Saïr, Robin Fuchs, Ambroise Coulomb, Enzo Terreau, Antoine Hoorelbeke*

## Introduction

PM2.5 refers to atmospheric particulate matter (PM) that have a diameter of less than 2.5 micrometers, which is about 3% the diameter of a human hair.

![pm25_comparison]

Since they are so small and light, fine particles tend to stay longer in the air than heavier particles. This increases the chances of humans and animals inhaling them into the bodies.

Studies have found a close link between exposure to fine particles and premature death from heart and lung disease. Fine particles are also known to trigger or worsen chronic disease such as asthma, heart attack, bronchitis and other respiratory problems.

## The problem

Since PM2.5 is such a public health issue, it would be very convenient to estimate and anticipate its level in the air. Thus, the data provided is composed of weather records measured in different cities in China. The target variable is the level of PM2.5 at the same time and same location.  

This [notebook](./Introduction.ipynb) is an exploratory analysis of the data available for this challenge. Also, there is a more detailled explanation of the problem, the business case and the evaluation.


## Set up

1. Install the ramp-workflow library (if not already done)  
``
$ pip install git+https://github.com/paris-saclay-cds/ramp-workflow.git
``

2. Follow the ramp-kits instructions from the [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki) 

The files [environement.yml](environement.yml) and [requirements.txt](requirements.txt) are useful to install the different depedencies required for the project.

## Starting kit

A first model has been designed in this [folder](./submissions/starting_kit). It is composed of a feature extractor that is performing preprocessing of the data and a regressor which implements the fit and predict methods.

To test the starting kit:  

``
$ ramp_test_submission --submission starting_kit
``

## Your submission

Add your own submission with the same structure as the starting kit in the folder submissions. Your FeatureExtractor and Regressor must implement the mandatory methods and inherit the same classes than the starting kit. Then, test your submission with the command:

``
$ ramp_test_submission --submission your_submission
``

Your model will be trained with a cross-validation process with 5 folds. On each fold, some metrics will be computed on the different sets.

## Help

Go to the [ramp-workflow wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki) for more help on the RAMP ecosystem.


## References

https://blissair.com/what-is-pm-2-5.html

[pm25_comparison]: ./img/pm25_comparison.jpg

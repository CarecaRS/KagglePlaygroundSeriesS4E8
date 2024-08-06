# Kaggle Playground Series
# S4E8 - Binary Prediction of Poisonous Mushrooms

The goal of this competition is to predict whether a mushroom is edible or poisonous based on its physical characteristics. Scores/evaluation given based on Matthew's correlation coefficient (MCC).

For practical use, I re-formatted the .csv files from Kaggle to .parquet, so GitHub could handle them.

## Features in this database
The feature `class` is a string, and it is what we need to predict, either a given mushroom is (p)oisonous or (e)dible

The features `cap-diameter`, `stem-height` and `stem-width` are all floats, measurement of such characteristics. All other features are strings that use the alphabet letters to attribute their variations.

## NaN values
Some features have too many NaN values, I'm dropping off the features with NaN over 50% of the sample. BTW, the NaN proportions are quite the same from both test and train datasets.

Numeric NaNs were imputed with the feature's median. Categorical NaNs were impute by a string 'nan', as a way to do a naive model as soon as possible and assess the results.

## My proposed solution
!IMPORTANT NOTICE!

This is still a work in progress. The .py file is not completed yet and may content some code that isn't quite right or optimal, it's just the way I like to work on my ideas.

!END OF IMPORTANT NOTICE!

I just got registered in this challenge, I'll code a simple model with XGB/CatBoost/LightGBM and assess the results. If the results aren't the best, I'm gonna code a neural network model, if there are enough observations for training. 

## Final score and classification
To be seen yet. Let's see what my models can do here.

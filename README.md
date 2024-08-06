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

First naive model using sklearn `HistGradientBoostingClassifier` scores 0.86xx locally. Adjusting some hyperparameters I managed to get to 0.97824 local score, which after submission granted a 0.97828 score on site. Great news that the MCC scores match.

I tried another two naive classification models, namely `CatBoostClassifier` and `XGBoostClassifier`, the former performing 0.9725 score and the latter just around 0.85xx. The actual scores and the models' parameters can bee seen in 'registros_resultados.csv' for the scores and in 'registros_modelagem.txt' for the parameters in each and every instance recorded, both within 'registros' directory.

I'm gonna try out some hyperparamenter tuning both in `CatBoostClassifier` and `HistGradientBoostingClassifier` in order to get a better score and maybe break the 0.98xx score threshold.

## Final score and classification
To be seen yet. Let's see what my models can do here.

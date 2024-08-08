# Kaggle Playground Series
# S4E8 - Binary Prediction of Poisonous Mushrooms

The goal of this competition is to predict whether a mushroom is edible or poisonous based on its physical characteristics. Scores/evaluation given based on Matthew's correlation coefficient (MCC).

For practical use, I re-formatted the .csv files from Kaggle to .parquet, so GitHub could handle them.

## My current goal
Break the 0.98xx score threshold with a good margin.

## Features in this database
The feature `class` is a string, and it is what we need to predict, either a given mushroom is (p)oisonous or (e)dible

The features `cap-diameter`, `stem-height` and `stem-width` are all floats, measurement of such characteristics. All other features are strings that use the alphabet letters to attribute their variations.

## NaN values
Some features have too many NaN values, I'm dropping off the features with NaN over 50% of the sample. BTW, the NaN proportions are quite the same from both test and train datasets.

Numeric NaNs were imputed with the feature's median.

Categorical NaNs were first imputed by a string 'nan', as a way to do a naive model as soon as possible. After the naive models were done, I proceed to estimate the categorical NaNs through classification models, as performed in `missing.py`. 

Since the NaN percentages in each feature between train/test sets, I started estimating the values by `gill-spacing`, followed by `cap-surface`, `gill-attachment` and `ring-type`. The features `cap-shape`, `cap-color`, `does-bruise-or-bleed`, `gill-color`, `stem-color`, `has-ring` and `habitat` had NaN observations which corresponded at most to ~0.002%from the feature pool, so I imputed the mode of each feature.

Accuracy score from the train set estimations, respectively: 0.995, 0.830, 0.963 and 0.982.

Accuracy score from the test set estimations, respectively: 0.993, 0.826, 0.963 and 0.982.

And the whole imputation thing returns nothing better than the previous models without estimating the correct imputations. Awesome.

## My proposed solution
!IMPORTANT NOTICE!

This is still a work in progress. The .py file is not completed yet and may content some code that isn't quite right or optimal, it's just the way I like to work on my ideas.

!END OF IMPORTANT NOTICE!

File `mushrooms_09804.py` contains the code for a score 0.9804 in the leaderboards using `CatBoost`. The MCC score returned by this model align almost perfectly with the leaderboard score, which is great. Since I haven't got any good improvement with this algorithm I'm keeping this code apart from the main one and gonna work on `HistGradientBoostingClassifier` (from sklearn) algorithm.

I did a naive model using `XGBoostClassifier`, but after some tweaks my score didn't got anywhere near the ones obtained with `CatBoost` or `HistGradientBoostingClassifier`, so no more XGB modelling.

After a few attempts in get better scoring with `HistGradientBoostingClassifier` I couldn't get any significant improvement (if any), so I'm dropping this algorithm too.

The records about the respective algorithm scores and the models' parameters can bee seen in 'registros_resultados.csv' for the scores and in 'registros_modelagem.txt' for the hyperparameters in each and every instance recorded, both within 'registros' directory.

My best guess is that correctly imputing NaN values will solve my scoring threshold, it's what I'm working on now.

## Final score and classification
To be seen yet. Let's see what my models can do here.

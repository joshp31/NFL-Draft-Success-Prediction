1st: Created combine datasets from profootball reference data from 2008-2020. Manually classified DL and OLB prospects as either EDGES, DTs, or LBs. Also classified DEs as EDGEs, ILBs as LBs. This was necessary as some OLBs play off the ball, while some rush the passer. Also some DLs are EDGEs and some are DTs.

in linear regression model, q-q plot shows extreme lack of normality -> TODO: transform the response

Performed Box-Cox transformation as their were significant normality issues and got Optimal lambda: 0.19729438412731226
this is close to 0, so a log transformation will be used

Best Adj R^2 non-transformed and transformed model is the full model without conference
I want to cross validate those models or perform some sort of training test split

Made models of 1 predictor vs. response to see which predictors have a relationship with 5 year AV (use non-transformed variable)
Statistical Significance in Predictors: 3cone (time and if done or not), 40yd (time only), arm_length (not significant), bench (not significant), broad jump (time and if done or not), forced_fumbles (significant), games_played (significant), hand_size (significant), Ht (not significant), sacks (significant), shuttle (time and if done or not), tackles (significant), tackles for loss (significant), vertical (time and if done or not), Wt (not significant) 

While many predictors are statistically significant individually, all small models exhibit low R² values, indicating that no single variable strongly explains variation in 5-year AV on its own.

Significance level is .05 for everything

Performed forward and backward selection on the transformed response, and both subset selection methods produced the same models.

Decision Tree and Random Forest models implemented with depth 4. Will be further tested when I get to training test splits and validation.

Did k-fold cv on some models that looked promising
10-fold cv done on: 3 different transformed linear regression models, forward selection (result for backward selection would be the same as both methods resulted in the same set of predictors), ridge, lasso
5 fold cv done on:


The log transformation of the response variable improved performance for linear models (not including shrinkage methods), but degraded performance for more flexible models like neural networks and random forests in this dataset. So when cv was performed, no log transformation was used on the random forest or neural network models or the lasso regression when cv was performed.

Last step: using best model (neural network) to predict this years draft class and output results

use model to predict this years draft class
Clean up readme and make linkedin post

(data from https://www.pro-football-reference.com/draft/2020-combine.htm) also college football reference
2nd: cleaned datasets (all players are players that attended the combine from 2008-2020 and were drafted)

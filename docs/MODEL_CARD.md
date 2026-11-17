# Model Card

## Intended use
Educational demonstration of a multiclass classification workflow and API deployment. Not intended for real insurance pricing, underwriting, eligibility, or financial decisions.

## Data
Supplied `insurance.csv`: 100 rows and 8 columns. Target: `insurance_premium_category`. The project uses BMI, age group, lifestyle risk, city tier, income, and occupation as predictors.

## Model
Scikit-learn Pipeline with a ColumnTransformer (OneHotEncoder for categorical variables and passthrough for numeric variables) followed by RandomForestClassifier.

## Evaluation design
80/20 random holdout split, followed by 5-fold shuffled KFold cross-validation on the training subset. The notebook outputs should be quoted exactly in any report. The small sample size means estimates have high uncertainty. KFold does not preserve class proportions in folds.

## Limitations and risks
Small dataset; potential sampling bias; feature rules are hand-engineered; impurity feature importance is not causal; class probabilities may not be calibrated; model performance may change across samples and populations. City tier mapping is a fixed hand-maintained list.

## Operational notes
API validates input types/ranges and returns model probabilities. Keep preprocessing logic identical between training and inference. Load only trusted pickle files and pin compatible dependencies.

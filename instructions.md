The goal is to investigate fairness in credit-risk classification using the German Credit dataset, with the main focus on age-related bias, since that is what Case Study 3 requires.
We will train logistic regression models to predict whether someone is a good or bad credit risk.
1. Baseline model
   Train a logistic regression using all relevant features, including age. This will be the main model we compare everything against.
2. Age-unaware model
   Train the same model, but remove age. This lets us test whether simply hiding age reduces differences between age groups.
3. Proxy-reduced model
   Train another model without age and possibly without variables that strongly act as proxies for age, such as employment, job type, housing, or credit history. We should justify these proxies based on the data rather than assume them beforehand.
Before training, we should first analyse the dataset itself. For example:
- how many people are in each age group
- whether younger and older applicants already have different proportions of good/bad credit labels
- whether important features are distributed differently across age groups
Then we compare the models on normal performance measures such as accuracy, precision, recall, F1-score and confusion matrices.
The main analysis should focus on fairness across age groups. We can compare:
- predicted good-credit rates / selection rates
- true-positive rates
- false-positive rates
- demographic parity difference
- equal opportunity difference
We also need to take the unequal error costs in the German Credit dataset into account, because falsely classifying a bad credit risk as good is considered more costly than the reverse. The dataset is also quite small, so we should report uncertainty or stability across different train/test splits.

One of the alternative models, or another method such as reweighting or threshold adjustment, should be treated as the fairness intervention. We then compare it with the baseline in terms of both predictive performance and fairness.

Gender can still be included as a secondary analysis, but the dataset does not contain a clean sex variable: sex is combined with personal status. So any gender conclusions need to be treated cautiously.
If the sample sizes are large enough, we could also do an intersectional analysis, for example comparing age groups together with the sex/personal-status variable.

Main research question could be:
“To what extent does a logistic regression credit-risk classifier show age-related disparities on the German Credit dataset, and can a fairness intervention reduce these disparities without causing unacceptable losses in predictive performance or increasing costly classification errors?”
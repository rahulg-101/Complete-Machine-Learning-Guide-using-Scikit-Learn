# Complete-Machine-Learning-Guide-using-Scikit-Learn-IITM-Machine-Learning-Practice-(MLP)

#### The repository covers a wide range of topics and algorithms required for different kinds of Machine Learning Tasks using the scikit-learn library
#### The course is curated by the Indian Institute of Technology - Madras (IITM)'s [BS Online Degree Course](https://study.iitm.ac.in/ds/) which basically covers everything to get you started with machine learning. You can also checkout their [Diploma Course](https://study.iitm.ac.in/diploma/) which covers the exact same topics but with shorter time duration and fewer courses
#### In the notebooks, I have personally added quite a few lines of explanation markdown statements that could be useful to understand the intrinsic nature of algorithms as to why we are using, what we are using etc.

> ### For those of you who are not very familiar with the mathematics behind algorithms, I recommend that when you start seeing the implementation of an algorithm, `YOU MUST UNDERSTAND HOW IT IS WORKING` and for that this playlist by Josh Starmer is an excellent one which teaches these topics in very easy to understand and graphical manner without loading you with lots of terminologies and numbers : 
> [Statquest with Josh Starmer](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF)

The course focuses on practical implementation of machine learning algorithm using scikit-learn APIs.

##### Running Examples

- California housing prediction for regression tasks. (From Notebook 1 to 4)
- MNIST Digit recognition for classification tasks (Notebook 5 onwards)

##### Table of contents
1. W1 - Overview of End To End Implementation (Regression Problems Start here)
    - Data loading
    - Basic data loading/generation features(load, fetch, make)
    - End to End Implementation of a Regression Problem

2. W2 - Data Preprocessing
    - Data preprocessing
    - Data cleaning
    - Feature Extraction - `DictVectorizer`
    - Data Imputation - `SimpleImputer`, `KNNImputer`
    - Feature Scaling - `MaxAbsoluteScaler`, `MinMaxScaler`, `StandardScaler`
    - Feature Transformation - `FunctionTransformer`
    - Polynomial Transformation - `PolynomialFeatures`
    - Discretization - `KBinsDiscretizer`
    - Handling categorical variables - `OrdinalEncoder`, `OneHotEncoder`, `LabelEncoder`, `MultiLabelBinarizer`, `pandas.get_dummies`, `add_dummy_feature`
    - Composite Transformers - `ColumnTransformer`, `TransformedTargetRegressor`
    - Feature Selection
        - Filter based methods - `VarianceThreshold`, `SelectKBest`, `SelectPercentile`, `GenericUnivariateSelect`
        - Wrapper based Methods - `RFE`, `SelectFromModel`, `SequentialFeatureSelector`
    - Feature extraction
        - PCA - `PCA` (I have mentioned in my notebook `(NOT NEED TO STUDY IN GREAT DETAIL RIGHT NOW)` but you guys should study this topic at the same time only because its very important and asked way too many times in interviews or some question always appear from this topic in exams and refer to the playlist if you are not able to understand it from the notebook because neither did I (Yeah !))
    - Pipeline - `Pipeline`, `make_pipeline`, `FeatureUnion`
    - Hyper Parameter tuning and Cross validation - `GridSearchCV`, `RandomizedSearchCV`
    - Handling imbalance(imblearn) - `RandomUnderSampler`, `RandomOverSampler`, `SMOTE`

3. W3 - Baseline Regression Models
    - Baseline models
    - How to build simple baseline models
    - Linear Regression
        - Normal equation method(`LinearRegression`)
        - Iterative optimisation method(`SGDRegressor`)

4. W4 - Regularized Regression and Hyperparameter Tuning
    - California Housing Prediction
    - Exploratory data Analysis
    - Regularised Linear regression and Hyper parameter tuning
    - `Polynomial`, `Ridge` & `Lasso` Regressions

5. W5 - Classification & Evaluation Metrics (Classification Problems Start here)
    - Perceptron
        - Binary Classification
        - Multiclass Classification
        - Evaluation Metrics - `Confusion Matrix`, `Precision`,`Recall`, `Precision & Recall Tradeoff`, `ROC Curve`

6. W6 - Logistic and Naive Bayes Classification
    - `Logistic regression`
    - `Naive Bayes models`

7. W7 - Classification Models and Large Scale ML
    - Training Large Scale ML Models **`(Separate Notebook)`**
        - Learning in batches(partial_fit())
        - Vectorization Techniques - `CountVectorizer`,`HashingVectorizer`
    
    - `K Nearest Neighbour` model **`(Separate Notebook)`**
        - Classification
        - Regression
    - `Softmax Regression`
    - `Support Vector Machines`

8. W8 - Trees and Ensemble Techniques
    - Decision Trees **`(8a)`**
      - `DecisonTreeClassifier`
      - `DecisonTreeRegressor`
    - Ensemble Methods - Bagging**`(8b)`**
        - `Bagging`
        - `RandomForest`
        - `Voting estimators` 
    - Ensemble Methods - Boosting **`(8c)`**
        - `AdaBoost`
        - `GradientBoosting`
        - `XGBoost`

9. W9 - Clustering
    - Clustering
        - `K Means`
        - `Heirarchical Agglomerative Clustering (HAC)`
     
10. W10 - Multi_Layered_Perceptron
    - Multilayer_perceptron_classifier

## Thank you! If you enjoyed the contents of this repository, please consider giving it a Star!

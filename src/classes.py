import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, roc_auc_score, classification_report, confusion_matrix, precision_recall_curve, auc, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
from xgboost import XGBClassifier
from scipy.stats import skew
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from itertools import product
import warnings
warnings.filterwarnings('ignore')

############################################################################################################################################################################################
## Classes definitions for fraud prediction solution
## Handles Data Preprocessing, Feature Engineering, Train/Test Split, Class Balancing, Model Selection and Training, Hyperparameter tuning, Automation, Model Evaluation and Visualisatioons
############################################################################################################################################################################################

################################################################################################################################
# 1. Preprocessor Class
################################################################################################################################
class Preprocessor:
    """ Performs preprocessing e.g. feature engineering, normalization, feature selection, etc.. """
    def __init__(self, df, 
                 id_columns = ['payment_id', 'client_id'],
                 datetime_columns= ['transaction_date'],
                 target = 'fraud',
                 skew_threshold=0.0,
                 apply_scaling=False,
                 scaling_method='standard',
                 apply_imputation=False,  
                 imputer_strategy='median',
                 correlation_threshold=0.0,
                 outlier_threshold=0.0,
                 low_variance_threshold=0.0,
                 transform_skewed_y=False,
                 categorical_encoding='passthrough'):  
        """
        Initialization of the Preprocessor class.
        Parameters:
        X (pd.DataFrame): Input feature set.
        y (pd.Series): y variable.
        skew_threshold (float): Threshold to identify skewed features. Features with skew > skew_threshold will be transformed. If 0.0, skew handling is skipped.
        scaling_method (str): Scaling method to apply, 'standard' for StandardScaler, 'minmax' for MinMaxScaler.
        imputer_strategy (str): imputer strategy (One of: mean, median, most_frequent, constant etc.)
        correlation_threshold (float): Threshold for identifying highly correlated features.
        outlier_threshold (float): Multiplier for IQR-based outlier detection. If 0.0, outlier removal is skipped.
        low_variance_threshold (float): Threshold for feature selection based on variance. Features with variance below this threshold will be removed. If 0.0, low variance removal is skipped.
        transform_skewed_y (float): Indicates if to transform a skewed target variable.
        categorical_encoding (str): Encoding type for categorical features ('onehot' or 'label').
        apply_imputation (bool): Whether to execute imputation on numerical and categorical features.
        apply_scaling (bool): Whether to execute scaling on numerical features.
        """
        self.X = df.drop(columns=[target])
        self.y = df[target]        
        self.skew_threshold = skew_threshold
        self.scaling_method = scaling_method
        self.imputer_strategy = imputer_strategy       
        self.correlation_threshold = correlation_threshold
        self.outlier_threshold = outlier_threshold
        self.low_variance_threshold = low_variance_threshold
        self.transform_skewed_y = transform_skewed_y
        self.column_transformer = None
        self.categorical_encoding = categorical_encoding
        self.apply_imputation = apply_imputation
        self.apply_scaling = apply_scaling
        self.id_columns = id_columns
        self.datetime_columns = datetime_columns      
        # Exclude id_columns and datetime_columns from columns removal
        self.columns_to_check = [col for col in self.X.columns if col not in self.id_columns and col not in self.datetime_columns]
        # Identify numerical and categorical features excluding id and datetime columns
        self.numerical_features = [col for col in self.X.select_dtypes(include=['float64', 'int64']).columns.tolist() if col not in self.id_columns and col not in self.datetime_columns]
        self.categorical_features = [col for col in self.X.select_dtypes(include=['object', 'category']).columns.tolist() if col not in self.id_columns and col not in self.datetime_columns]
        
        # Convert transaction_date column to datetime format
        for col in datetime_columns:
            self.X[col] = pd.to_datetime(self.X[col], errors='coerce')

        # Remove target column from numerical features if it exists
        if self.y.name in self.numerical_features:
            self.numerical_features.remove(self.y.name)

    
    def execute_preprocessing_pipeline(self):
        """
        Executes all preprocessing steps sequentially on the dataset (self.X and self.y)
        Applies imputation, skew transformation, scaling, correlations handling, outliers removal and low variance filtering if needed.

        Returns:
        X_transformed (pd.DataFrame): Fully preprocessed feature set.
        y_transformed (pd.Series): Fully preprocessed y variable.
        """
        print("\n\n[DEBUG] Starting full preprocessing pipeline execution...")

        # First, apply imputation and scaling
        # Impute and scale numerical features
        X_transformed = self.X.copy()
        y_transformed = self.y.copy()

        if self.apply_imputation:
            # Apply imputations
            num_imputer = SimpleImputer(strategy=self.imputer_strategy)
            cat_imputer = SimpleImputer(strategy='most_frequent')
            # Impute numerical features
            X_transformed[self.numerical_features] = num_imputer.fit_transform(X_transformed[self.numerical_features])
            # Impute categorical features
            X_transformed[self.categorical_features] = cat_imputer.fit_transform(X_transformed[self.categorical_features])
            print("[DEBUG] Imputation applied.")
        else:
            print("[DEBUG] Imputation skipped as apply_imputation is False.")

        if self.apply_scaling:
            # Apply scaling
            scaler = StandardScaler() if self.scaling_method == 'standard' else MinMaxScaler()
            X_transformed[self.numerical_features] = scaler.fit_transform(X_transformed[self.numerical_features])
            print("[DEBUG] Scaling applied.")
        else:
            print("[DEBUG] Scaling skipped as apply_scaling is False.")

        # Categorical encoding 
        categorical_columns = [col for col in self.categorical_features if col not in self.datetime_columns]
        if self.categorical_encoding == 'label': # apply Label Encoding
            for feature in categorical_columns:
                print(f"[DEBUG] Applying Label Encoding to {feature}...")
                le = LabelEncoder()
                X_transformed[feature] = le.fit_transform(X_transformed[feature].astype(str))
        if self.categorical_encoding == 'onehot': # If apply onehot encoding
            self.column_transformer = ColumnTransformer(
                transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)])
            # Fit the transformer on training data
            self.column_transformer.fit(X_transformed)
            # Apply the transformation
            X_transformed = self.column_transformer.transform(X_transformed)
            # Get feature names from the OneHotEncoder
            cat_feature_names = self.column_transformer.transformers_[0][1].get_feature_names_out(categorical_columns)
            # Convert the sparse matrix to a DataFrame with the correct column names
            X_transformed = pd.DataFrame(X_transformed.toarray(), columns=cat_feature_names)
                               
        # Optionally removes features with low variance.         
        if self.low_variance_threshold > 0:
            X_transformed = self.remove_low_variance_features(X_transformed)
        else:
            print("[DEBUG] Low variance removal skipped as low_variance_threshold is 0.0.")

        # handle_correlations between features.        
        if self.correlation_threshold > 0:
            X_transformed = self.handle_correlations(X_transformed)
        else:
            print("[DEBUG] Correlations handling skipped as correlation_threshold is 0.0.")

        # Optionally remove rows with outliers.         
        if self.outlier_threshold > 0:
            X_transformed = self.remove_outliers(X_transformed)
        else:
            print("[DEBUG] Outlier removal skipped as outlier_threshold is 0.0.")

        # Transform skewed features using a log transformation.
        if self.skew_threshold > 0:
            X_transformed = self.transform_skewed_features(X_transformed)
        else:
            print("[DEBUG] Skew handling skipped as skew_threshold is 0.0.")

        # Optionally Transform skewed y using a log transformation.      
        if self.transform_skewed_y == True:
            y_transformed = self.preprocess_y()
        else:
            print("[DEBUG] skewed y transformation skipped as transform_skewed_y is False.")

        # At the end, add hour, month and day_of_week features for all datetime columns and remove the originals.
        X_transformed = self.add_datetime_features(X_transformed)
            
        print("[DEBUG] Preprocessing complete!\n\n")
        return X_transformed, y_transformed

    
    def transform_skewed_features(self, X_transformed):
        """ Applies log1p transformation (log(x+1)) to any skewed numerical features in the dataset."""
        # Identify skewed features (numerical only)
        skewed_features = [feature for feature in self.numerical_features if feature in X_transformed.columns and skew(self.X[feature].dropna()) > self.skew_threshold]
        print(f"\n\n[DEBUG] Transforming skewed features using log1p with skew_threshold {self.skew_threshold}...: {skewed_features}")
        
        for feature in skewed_features:
            X_transformed[feature] = np.log1p(X_transformed[feature])
            
        return X_transformed

    
    def remove_outliers(self, X_transformed):
        """Detect and remove rows with outliers from the X using the IQR (Interquartile Range) method based on the specified threshold"""
        check_features = [feature for feature in self.columns_to_check if feature in X_transformed.columns] ## In case some features were already removed
        Q1 = X_transformed[check_features].quantile(0.25)  # 25th percentile
        Q3 = X_transformed[check_features].quantile(0.75)  # 75th percentile
        IQR = Q3 - Q1
    
        # Determine outliers
        is_outlier = (
            (X_transformed[check_features] < (Q1 - self.outlier_threshold * IQR)) | (X_transformed[check_features] > (Q3 + self.outlier_threshold * IQR))
        ).any(axis=1)

        print(f"[DEBUG] Number of outliers detected with outlier_threshold {self.outlier_threshold}...: {is_outlier.sum()}")
        return X_transformed[~is_outlier]  # Rows with is_outlier == True are removed, and only rows without outliers are returned.

    
    def handle_correlations(self, X_transformed):
        """Identify highly correlated features and remove features that have a high correlation with other features (based on the specified correlation threshold) to reduce multicollinearity.
           Excluding id_columns and datetime_columns.
        """
        check_features = [feature for feature in self.columns_to_check if feature in X_transformed.columns] ## In case some features were already removed
        corr_matrix = X_transformed[check_features].corr().abs()  # Abs value of correlation coefficients between each pair of features in the dataset
        #The top right part of the correlation matrix contains the correlation values between each pair of features. The bottom left part and the diagonal are filled with NaN.
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > self.correlation_threshold)]
        print(f"\n\n[DEBUG] Removing correlated features with correlation higher than {self.correlation_threshold}...: {to_drop}")
        
        return X_transformed.drop(columns=to_drop, axis=1)

        
    def remove_low_variance_features(self, X_transformed):
        """
        Remove features with variance below the specified threshold, excluding id_columns and datetime_columns.
        """        
        check_features = [feature for feature in self.columns_to_check if feature in X_transformed.columns] ## In case some features were already removed
        selector = VarianceThreshold(threshold=self.low_variance_threshold) # removes all features with variance below the specified threshold.
        selector.fit(X_transformed[check_features]) # calculates the variance for each feature in X_transformed.

        # Identify low-variance features
        low_variance_features = [feature for feature, support in zip(check_features, selector.get_support()) if not support]
    
        print(f"\n\n[DEBUG] Removing features with variance below {self.low_variance_threshold}...: {low_variance_features}")
    
        # Filter columns to retain all features 
        retained_features = self.id_columns + self.datetime_columns + [feature for feature in check_features if feature not in low_variance_features]
        return X_transformed[retained_features]  

    
    def preprocess_y(self):
        """
        Preprocess the y variable (self.y)  by applying log1p transformation if skewed.
        Returns:
        (pd.Series): Transformed y variable.
        """
        if skew(self.y) > self.skew_threshold:
            print(f"\n\n[DEBUG] y is skewed. Applying log1p transformation...")
            return np.log1p(self.y)

        print("[DEBUG] y is not skewed. No transformation applied.")        
        return self.y
        
    def _get_feature_names(self):
        """Get feature names after transformation."""
        if self.categorical_encoding == 'onehot':
            cat_features = list(self.column_transformer.transformers_[0][1].get_feature_names_out(self.categorical_features))
            return self.numerical_features + cat_features
            
        return self.numerical_features + self.categorical_features

    def add_datetime_features(self, X_transformed):
        """
        Extracts 'hour', 'day_of_week', and 'month' from all datetime columns
        and drops the original datetime columns.
        """
        datetime_columns = X_transformed.select_dtypes(include=['datetime64[ns]', 'datetime64', 'datetime']).columns
        
        for col in datetime_columns:
            print(f"[DEBUG] Extracting datetime features for column: {col}")
            X_transformed[f"{col}_hour"] = X_transformed[col].dt.hour
            X_transformed[f"{col}_day_of_week"] = X_transformed[col].dt.dayofweek
            X_transformed[f"{col}_month"] = X_transformed[col].dt.month
            
        # Drop the original datetime columns
        return X_transformed.drop(columns=datetime_columns, axis=1)


################################################################################################################################
# 2. DatasetSplitter Class
################################################################################################################################
class DatasetSplitter:
    """
    Split the full dataset into train and test ensuring that test will contain approx input percentage
    Guarantee that test and train sets contain totally different client_ids and months
    Test months are selected as the latest months (out-of-time split).
    """
    def __init__(self, X, y, test_percentage, client_id_col='client_id',month_col='transaction_date_month'):
        """
        Initialize the DatasetSplitter.
        Parameters:
        - X (pd.DataFrame): Feature dataset.
        - y (pd.Series): Target labels.
        - client_id_col (str): The name of the column representing the client IDs.
        - month_col (str): The name of the column containing the transaction months.
        - test_percentage (float): The approximate percentage of data to allocate to the test set.
        """
        self.X = X
        self.y = y
        self.client_id_col = client_id_col
        self.month_col = month_col
        self.test_percentage = test_percentage

    def split(self):
        """
        - Split data into training, and testing sets ensuring no data leakage of clients and months. 
        - The test set contains approximately the specified percentage of the data.
        - Test months are selected as the latest months (out-of-time split).

        Returns:
        - X_train (pd.DataFrame): The training feature set.
        - X_test (pd.DataFrame): The testing feature set.
        - y_train (pd.Series): The training labels.
        - y_test (pd.Series): The testing labels.
        """
        # Combine X and y for splitting
        data = self.X.copy()
        data['target'] = self.y

        # Ensure the month column is sorted numerically
        data = data.sort_values(by=self.month_col)

        # Calculate cumulative size per month to find the cutoff for the test set
        monthly_counts = data.groupby(self.month_col).size().cumsum()
        total_size = len(data)
        test_size = int(total_size * self.test_percentage)

        # Identify the cutoff month where the test set starts
        cutoff_month = monthly_counts[monthly_counts >= (total_size - test_size)].index[0]

        # Split into train and test based on the cutoff month
        test_data = data[data[self.month_col] >= cutoff_month]
        train_data = data[data[self.month_col] < cutoff_month]

        # Ensure test and train sets have distinct client IDs
        test_client_ids = set(test_data[self.client_id_col])
        train_data = train_data[~train_data[self.client_id_col].isin(test_client_ids)]

        # Separate features and target for train and test sets
        X_train = train_data.drop(columns=['target'])
        y_train = train_data['target']
        X_test = test_data.drop(columns=['target'])
        y_test = test_data['target']

        print(f"\n\n[DEBUG] Train size: {len(X_train)}, Test size: {len(X_test)}")
        print(f"[DEBUG] Test set contains {len(test_client_ids)} unique client_ids.")
        print(f"[DEBUG] Test starts at month: {cutoff_month}")

        return X_train, X_test, y_train, y_test


################################################################################################################################
# 3. ClassBalancer Class
################################################################################################################################
class ClassBalancer:
    def __init__(self, method='upsample', target_ratio=1.0):
        """
        Handle class imbalance using up-sampling or down-sampling.

        Parameters:
        method (str): Sampling method ('upsample' or 'downsample').
        target_ratio (float): Desired fraud/non-fraud ratio.
        """
        self.method = method
        self.target_ratio = target_ratio

    def balance(self, X, y):
        """
        Balances the dataset classes according to the specified method and ratio.
        
        Parameters:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target data.
        
        Returns:
        X_resampled (pd.DataFrame): Resampled feature data.
        y_resampled (pd.Series): Resampled target data.
        """
        print(f"\n\n[DEBUG] Balancing classes with method: {self.method}, target ratio: {self.target_ratio}...")
        if self.method == 'upsample':
            sampler = RandomOverSampler(sampling_strategy=self.target_ratio)
        elif self.method == 'downsample':
            sampler = RandomUnderSampler(sampling_strategy=self.target_ratio)
        else:
            print(f"[DEBUG] Balancing skipped because method is: {self.method}", f"to balance use 'upsample' or 'downsample'.")
            return X, y
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        print(f"[DEBUG] Resampled dataset: {dict(pd.Series(y_resampled).value_counts())}\n\n")
        
        return X_resampled, y_resampled


################################################################################################################################
# 4. Model Class
################################################################################################################################
class Model:
    """
    A generic machine learning model class that allows flexible training, prediction, and loss calculation on any dataset provided and any machine learning model that follows the scikit-learn API along with its hyperparameters.

    Attributes:
    -----------
    model : object - The machine learning model object (e.g., RandomForestClassifier, XGBClassifier)
    """
            
    def __init__(self, model_class, X_train, y_train, model_params=None, id_columns=None):
        """
        Initialize the Model class with a specific model, features, target and parameters.
        Parameters:
        - model_class : object - The machine learning model to be used (e.g., RandomForestClassifier, XGBClassifier).
        - model_params : dict - Hyperparameters for the model, to be passed during initialization.
        - X_train: Training feature set
        - y_train: Training labels        
        """
        self.trained_model = None
        self.id_columns = id_columns or []
        # Drop id_columns from the training data
        self.X_train = X_train.drop(columns=self.id_columns, axis=1, errors='ignore')
        self.y_train = y_train
         # Initialize the model with a scikit-learn compatible mode and external hyperparameters
        self.model = model_class(**model_params) if model_params else model_class()
        print(f"Model initialized with parameters: {model_params}")
    
    def train(self):
        """
        Train the model to the training data (X_train, y_train) By wrapping the `fit` method of the underlying machine learning model.
        Parameters:
        """
        print("\n\n[DEBUG] Fitting the model...")
        self.trained_model = self.model.fit(self.X_train, self.y_train)
        print(f"Model {self.trained_model.__class__.__name__} fitting completed.")

        return self.trained_model
   
    def eval_report(self, y_pred, y_pred_proba, y_test=None, Plot_ROC_curve = False, print_Classification_Report = False):
        """
        Evaluate the model on the eval data and print key metrics.
        Parameters:
        y_test: True labels        
        y_pred: predicted labels        
        y_pred_proba: predicted labels probabilities      
        """           
        if y_test is None:
            raise ValueError("True labels not provided")

        print("[INFO] Evaluating classifier...")
        if print_Classification_Report:
            print("Classification Report:\n", classification_report(y_test, y_pred))
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
            
        # ROC AUC Score
        roc_auc_score1 = roc_auc_score(y_test, y_pred_proba)        
        print(f"ROC AUC Score: {roc_auc_score1:.4f}")
        
        # Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        if Plot_ROC_curve == True:
            print(f"PR-AUC: {pr_auc}\n")            
            # Plot the ROC curve
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, label='Random guess')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.show()


        return {
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "roc_auc": roc_auc,
            "roc_auc_score": roc_auc_score1,
            "Precision-Recall AUC": pr_auc
        }
       
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the eval data and print key metrics.
        Parameters:
        X_test: eval feature set
        y_test: eval labels        
        """
        if self.trained_model is None:
            raise ValueError("Model has not been trained. Call train() before evaluate().")

        # Drop id_columns from the test data if exists
        X_test = X_test.drop(columns=self.id_columns, axis=1, errors='ignore')
        print("\n\n[DEBUG] Evaluating model...")
        y_pred = self.trained_model.predict(X_test)
        y_pred_proba = self.trained_model.predict_proba(X_test)[:, 1]

        return self.eval_report(y_pred, y_pred_proba, y_test)
    
    
    def predict(self, X):
        """
        Make predictions on new data.
        Parameters:
        X: Feature set for prediction -  pd.dataFrame or np.ndarray - 
        Returns:
        - np.ndarray - Predicted y values.        """
        if self.trained_model is None:
            raise ValueError("Model has not been trained. Call train() before predict().")
        X = X.drop(columns=self.id_columns, axis=1, errors='ignore')
        return self.trained_model.predict(X)
  
    
    def plot_feature_importance(self):

        """
        Get feature importance from the trained model for evaluating feature's relevancy and plot feature importance.
        """
        if not (hasattr(self.model, 'feature_importances_')):
            raise AttributeError(f"The model of type '{self.model_type}' does not have feature importances.")

        feature_importances = self.trained_model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)   
 
        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='blue')
        plt.xlabel('Feature Importance')
        plt.ylabel('Features')
        plt.title(f"\nFeature Importance from {self.trained_model.__class__.__name__}")
        plt.gca().invert_yaxis()  # Invert axis to show most important at the top
        plt.show()

        return importance_df 

    def fit_and_evaluate(self, X_test, y_test): 
        """ Helper function to Train and evaluate classifier """
        # Fit the model on the training X
        trained_model = self.train()
    
        # Predictions on the test set 
        predictions = self.evaluate(X_test, y_test)
        
        # Get feature importance
        importance_df = self.plot_feature_importance()
            
        return trained_model,importance_df, predictions


################################################################################################################################
# 5. HyperparameterTuner Class
################################################################################################################################
class HyperparameterTuner:
    """ 
    Perform hyperparameter tuning using `GridSearchCV` 
    Generic and works with any model class that follows the scikit-learn API, such as `RandomForestRegressor`, `XGBRegressor`, etc.
    """
    def __init__(self, model_class, X_train, y_train, param_ranges, scoring='roc_auc', n_jobs=-1):
        """
        Initialize the HyperparameterTuner class.
        Parameters:
        - model_class: The model class (e.g., RandomForestRegressor, XGBRegressor).
        - X_train: Training feature set.
        - y_train: Training y variable.
        - param_ranges: A dictionary containing hyperparameter ranges to search over.
        - scoring: Scoring metric for model evaluation (default is 'neg_mean_squared_error' as grid Search try to maximize a score to find the best hyperparameters. Since MSE is a measure of error (to be minimized), it is converted to negative MSE so that maximizing it leads to finding the minimum error.
        - n_jobs: Number of parallel jobs for GridSearchCV (default is -1, using all processors).
        """
        self.model_class = model_class
        self.X_train = X_train
        self.y_train = y_train
        self.param_ranges = param_ranges
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.best_estimator = None
        self.best_params = None

    def tune(self):
        """
        Perform grid search using GridSearchCV to find the best hyperparameters.
        Returns:
        - best_params: The hyperparameters that resulted in the best performance.
        """
        # Instantiate the model (as required by GridSearchCV)
        base_model = self.model_class()

        # Set up GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model, 
            param_grid=self.param_ranges, 
            scoring=self.scoring, 
            n_jobs=self.n_jobs,
            cv=2,
            verbose=1
        )

        # Fit GridSearchCV grid search to find the best hyperparameters
        print("\n\n[DEBUG] HyperparameterTuner - Starting GridSearchCV...")
        grid_search.fit(self.X_train, self.y_train)

        # Store the best hyperparameters
        self.best_params = grid_search.best_params_
        self.best_estimator = grid_search.best_estimator_
        print(f"Best hyperparameters: {self.best_params}")

        return self.best_params, self.best_estimator

    
    # Create an instance of model with the optimal hyperparameters found.
    def best_model_instance(self, X_test, y_test):
        """ 
        Function to create an instance of a model with optimal hyperparameters  
        Parameters:
        - X_test: Test features.
        - y_test: Test labels.

        Returns:
        - Best model's predictions and evaluation metrics.
        """
        # Run grid search to tune hyperparameters for RandomForest
        best_params, best_estimator = self.tune()
        
        # Initialize a new best `Model` instance with the best hyperparameters.
        best_model = Model(self.model_class, self.X_train, self.y_train, model_params=best_params, id_columns = ['payment_id', 'client_id'])

        # Train the best model on the training set and evaluate the best model on the validation and test set
        best_trained_model, importance_df, predictions = best_model.fit_and_evaluate(X_test, y_test)
    
        print(f"Hyperparameters of best {best_trained_model.__class__.__name__}")
        for param, value in best_trained_model.get_params().items():
            print(f"{param}: {value}")
    
        return best_model, best_trained_model, importance_df, predictions


################################################################################################################################
# 6. WorkflowParameterTuner Class
################################################################################################################################
class WorkflowParameterTuner:
    """
    Comprehensive Tuning: Explores all combinations of hyperparameters across preprocessing, resampling, and modeling steps.
    Performance Metric: Maximizes the selected  (ROC AUC) score.
    Modular: Supports extending to other classifiers or stages with minor modifications.
    
    Returns:
        best_params, best_score, best_model
    """
    def __init__(self, 
                 preprocessor_param_ranges,
                 class_balancer_param_ranges,
                 model_param_ranges,
                 model_classes,
                 test_percentage,
                 scoring='roc_auc'):
        """
        Initializes the WorkflowParameterTuner.
        
        Parameters:
        - preprocessor_param_ranges (dict): Parameter ranges for the Preprocessor.
        - class_balancer_param_ranges (dict): Parameter ranges for ClassBalancer.
        - model_param_ranges (dict): Parameter ranges for the models.
        - model_classes (list): List of model classes to evaluate.
        - test_percentage (float): Test data percentage for splitting, parameter to DatasetSplitter
        - scoring (str): Scoring metric for evaluating model performance.
        """
        self.preprocessor_param_ranges = preprocessor_param_ranges
        self.class_balancer_param_ranges = class_balancer_param_ranges
        self.model_param_ranges = model_param_ranges
        self.model_classes = model_classes
        self.scoring = scoring
        self.test_percentage = test_percentage

    def _grid_search_params(self, param_ranges):
        """
        Generates all combinations of parameter values from the given ranges.
        """
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        for combination in product(*values):
            yield dict(zip(keys, combination))

    def tune(self, df, target_col='fraud', id_columns=['payment_id', 'client_id'], datetime_columns=['transaction_date']):
        """
        Executes the automated pipeline to find the best parameter combination.        
        Parameters:
        - df (DataFrame): The input data.
        - target_col (str): The target column in the dataset.
        - id_columns (list): List of ID columns to exclude from feature processing.
        - datetime_columns (list): List of datetime columns for preprocessing.

        Returns:
        - best_params (dict): The best parameter combination.
        - best_score (float): The ROC AUC score of the best parameter combination.
        - best_model (object): The trained model using the best parameters.
        """
        best_score = -float('inf')
        best_params = {}
        best_model = None

        for preprocessor_params in self._grid_search_params(self.preprocessor_param_ranges):
            # Step 1: Preprocessing
            preprocessor = Preprocessor(
                df=df, 
                target=target_col, 
                id_columns=id_columns, 
                datetime_columns=datetime_columns, 
                **preprocessor_params
            )
            X_preprocessed, y_preprocessed = preprocessor.execute_preprocessing_pipeline()

            # Step 2: Splitting data
            splitter = DatasetSplitter(
                X=X_preprocessed, 
                y=y_preprocessed, 
                test_percentage = self.test_percentage
            )
            X_train, X_test, y_train, y_test = splitter.split()

            for balancer_params in self._grid_search_params(self.class_balancer_param_ranges):
                # Step 3: Balancing classes
                class_balancer = ClassBalancer(**balancer_params)
                X_train_balanced, y_train_balanced = class_balancer.balance(X_train, y_train)

                for model_class in self.model_classes:
                    for model_params in self._grid_search_params(self.model_param_ranges):
                        # Step 4: Training the model
                        model = Model(
                            model_class=model_class, 
                            X_train=X_train_balanced, 
                            y_train=y_train_balanced, 
                            model_params=model_params, 
                            id_columns=id_columns
                        )
                        trained_model = model.train()
                        print(f"\nModel: {trained_model.__class__.__name__}")

                        # Step 5: Evaluate the model
                        predictions = model.evaluate(X_test, y_test)
                        score = predictions['roc_auc_score']
                                
                        # Update the best parameters if the current score is better
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'preprocessor': preprocessor_params,
                                'class_balancer': balancer_params,
                                'model_class': model_class,
                                'model': model_params
                            }
                            best_model = model

        print(f"Best ROC AUC Score: {best_score}")
        print(f"Best Parameters: {best_params}")
        return best_params, best_score, best_model, trained_model


################################################################################################################################
# 7. Visualizations_and_explorations Class
################################################################################################################################
class Visualizations_and_explorations():
    """
    A class used to visualize various aspects of a dataset using:
    - Correlation heatmaps
    - Feature distributions
    - Pair plots between features and y
    - Box plots for categorical variables
    """

    def __init__(self, df, id_columns, datetime_columns, target = 'fraud'):
        """
        Initialize the Visualization class with a X and y variables.
        
        Parameters:
        -----------
        X : The dataFrame containing feature X.
        y : The Series containing the y variable.
        """
        self.df = df
        self.X = df.drop(columns=[target]).drop(columns=id_columns, errors='ignore')
        self.y = df[target]
        self.numeric_features = [col for col in self.X.select_dtypes(include=['float64', 'int64']).columns.tolist() ]
        self.categorical_columns = [col for col in self.X.select_dtypes(include=['object']).columns.tolist() ]
        # Convert transaction_date column to datetime format
        for col in datetime_columns:
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

    
    def basic_data_exploration(self):
        """
        Performs initial X exploration, providing insights into the structure, types, statistics, and possible preprocessing steps.
        Parameters:
        - df: The full pandas DataFrame to be analyzed. 
        - X: Features 
        - y: target variable 
        Output:
        - Prints summary information about the dataset.
        """
        # Check the shape of the dataset (rows, columns)
        print("\n\nShape of dataset (df.shape):\n", self.df.shape, "\n")
    
        # dataFrame structure: Look at first few rows
        print("First 5 rows of the dataFrame:\n", self.df.head(), "\n")
        
        # Check X types, number of non-null values, and memory usage.
        print("\nX Information (df.info()):\n")
        print(self.df.info(), "\n")
        
        # Check for missing values to decide whether imputation or dropping missing X will be necessary.
        print("\nMissing Values (df.isnull().sum()):\n", self.df.isnull().sum(), "\n")
    
        # Summary statistics (mean, std, min, max, etc.) and check if features have varying scales (important for normalization)
        print("\nSummary Statistics (df.describe()):\n", self.df.describe(), "\n")
    
        # Check for duplicates
        print("\nNumber of duplicate rows (df.duplicated().sum()):\n", self.df.duplicated().sum(), "\n")      
        
        # Displays the number of unique values in each column to check for potential categorical features or features with imbalance
        unique_values_per_column = self.df.nunique()
        print("Number of unique values per column:\n", unique_values_per_column, "\n")
    
        # Check for features with a single unique value (constant columns), which may not contribute to the model.
        constant_columns = unique_values_per_column[unique_values_per_column == 1]
        if not constant_columns.empty:
            print("\nConstant columns with a single unique value\n",constant_columns, "\n")
        else:
            print("\nNo constant columns with a single unique value found.\n")
    
        # Return information about the dataset
        print("\nBasic data exploration complete.\n")
        
    def correlation_heatmap(self, threshold):
        """
        Plot a correlation heatmap for the features in the dataset to visualizes the pairwise correlation of features using a heatmap, highlighting strong correlations
        """
        # Compute correlation matrix for numeric features only
        # If a y variable is provided, show correlation of features with the y.
        if self.y is not None:
            X_with_y = self.X[self.numeric_features].copy()
            X_with_y['y'] = self.y
            corr_matrix = X_with_y.corr()
        else:
            corr_matrix = self.X[self.numeric_features].corr()
        
        # Plot the heatmap
        plt.figure(figsize=(20, 16))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
        plt.title("Correlation Heatmap for Numeric Features")
        plt.show()

        # Find high-correlation pairs
        print(f"[DEBUG] Checking correlations above {threshold}...")
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_pairs = [(col, row, corr_matrix.loc[row, col]) 
                       for col in upper_triangle.columns 
                       for row in upper_triangle.index 
                       if abs(corr_matrix.loc[row, col]) > threshold and abs(corr_matrix.loc[row, col]) <1]
    
        if high_corr_pairs:
            print("[WARNING] High-correlation pairs detected:")
            for pair in high_corr_pairs:
                print(f"  {pair[0]} and {pair[1]}: {pair[2]:.2f}")
        else:
            print("[INFO] No high correlations found above the threshold.")


    def feature_distribution(self, skew_threshold = 1.5):
        """
        Plot the distribution of each numerical feature in the dataset using histograms and KDE (Kernel Density Estimate) plots. 
        """
        # Check for right-skewed features (tail on the right side) 
        print("\nRight skewed cols:\n", self.X[self.numeric_features].skew().index[self.X[self.numeric_features].skew() > skew_threshold], "\n")
    
        # Check for left-skewed features (tail on the left side) 
        print("\nLeft skewed cols:\n", self.X[self.numeric_features].skew().index[self.X[self.numeric_features].skew() <= -(skew_threshold)], "\n\n")
    
        for column in self.X.columns:
            if not (np.issubdtype(self.X[column].dtype, np.datetime64) or isinstance(self.X[column].dtype, pd.PeriodDtype)):
                #print(column,"\n", self.X[column].dtype)               
                plt.figure(figsize=(5, 3))
                sns.histplot(self.X[column], kde=True, bins=30, color='blue')
                plt.title(f'Distribution of {column}')
                plt.xlabel(column)
                plt.ylabel('Frequency')
                plt.show()

        #If the y variable is provided, display it separately.
        if self.y is not None:
            print('Distribution of y Variable\n')    
            plt.figure(figsize=(10, 6))
            sns.histplot(self.y, kde=True, bins=30, color='orange')
            plt.title('Distribution of y Variable')
            plt.xlabel('y')
            plt.ylabel('Frequency')
            plt.show()

    def pair_plot(self):
        """
        Create a pair plot between all features and the y variable.
        This method visualizes the relationships between features using scatter plots and KDE plots.
        """
        if self.y is not None:
            X_with_y = self.X[self.numeric_features].copy()
            X_with_y['y'] = self.y
            sns.pairplot(X_with_y, diag_kind="kde")
        else:
            sns.pairplot(self.X[self.numeric_features], diag_kind="kde")
        plt.show()

    def box_plot(self):
        """
        Plot box plots for categorical variables against the y variable. (If categorical variables exist)
        visualize how different categories (if any) relate to the y variable.
        """
        if self.y is None:
            print("y variable not provided for box plot.")
            return
        
        for column in self.categorical_columns:
            if column in self.X.columns and not (np.issubdtype(self.X[column].dtype, np.datetime64) or isinstance(self.X[column].dtype, pd.PeriodDtype)):
                #print(column,"\n", self.X[column].dtype)    
                plt.figure(figsize=(5, 3))
                sns.boxplot(x=self.X[column], y=self.y)
                plt.title(f'Box Plot of {column} vs y')
                plt.xlabel(column)
                plt.ylabel('y')
                plt.show()
            else:
                print(f"Column {column} not found in X.")


    def target_vs_features(self):
        """
        Plot the y variable against each feature individually.      
        Create scatter plots of each numerical feature against the y variable, allowing to observe the relationships with more details.
        """
        if self.y is None:
            print("y variable not provided.")
            return

        # Calculate the correlation matrix between each feature and the y to get insights into feature importance.
        print("\nCorrelation of features with y variable :\n", self.X[self.numeric_features].corrwith(self.y))

        for column in self.X.columns:
            if np.issubdtype(self.X[column].dtype, np.number) and not (np.issubdtype(self.X[column].dtype, np.datetime64) or isinstance(self.X[column].dtype, pd.PeriodDtype)):
                #print(column,"\n", self.X[column].dtype)    
                plt.figure(figsize=(5, 3))
                sns.scatterplot(x=self.X[column], y=self.y)
                plt.title(f'y vs {column}')
                plt.xlabel(column)
                plt.ylabel('y')
                plt.show()

    def plot_prediction_vs_actual(self, y_true, y_pred, title="Prediction vs Actual", xlabel="Actual Values", ylabel="Predicted Values"):
        """tarting full preprocessing pipeline executio
        Helper function for plotting a scatter plot of the predicted values (`y_pred`) vs the actual values (`y_true`). 
        - Can be used for any regression models
        - Helps to see how well the model predictions match the actual X.
        Parameters:
        y_true (array-like): Actual y values.
        y_pred (array-like): Predicted y values from the model.
        title (str): The title of the plot.
        xlabel (str): Label for the x-axis (actual values).
        ylabel (str): Label for the y-axis (predicted values).
        """
        plt.figure(figsize=(8, 6))
        
        # Scatter plot
        plt.scatter(y_true, y_pred, color='blue', alpha=0.6, label="Predictions")
        
        # Plot a dashed 45-degree red line representing where perfect predictions would lie (where `y_true == y_pred`)
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label="Perfect Prediction")
        
        # Add labels, title, and legend
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        
        # Show the plot
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def run_explorations(self):
        """ Run visualizations and explorations """
        self.basic_data_exploration()
        
        # histplot the distribution of all the features and the y 
        self.feature_distribution()
        
        # Correlation Heatmap to visualize the correlation between features using a heatmap, highlighting strong correlations
        self.correlation_heatmap(threshold=0.9)
        
        # y vs. Features: For numeric features visualization to show how the target variable relates to each numeric feature.
        self.target_vs_features()
        
        # y vs. Features: For categoric features visualization to show how the target variable relates to each categoric feature.
        self.box_plot()
        
################################################################################################################################
# 8. Utils Class
################################################################################################################################
class Utils():
    """
    A class used for utilities:
    - load_data
    """

    def __init__(self):
        """
        Initialize the Utils class        
        """
        pass

    def load_data(self, data_file):
        """
        Parameters:
        -----------
        data_file : The data file to load.

        Returns:
        -----------  
        The output data frame in pandas format
        """
        # Load the dataset as a pandas df 
        df = pd.read_csv(data_file)
        df = df.drop(columns=['Unnamed: 0'], errors='ignore')
        return df

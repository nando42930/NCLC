from sklearn.preprocessing import LabelEncoder
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score


def analyze_relationship(df, r2_threshold, accuracy_threshold):
    """
    Evaluates predictive relationships between combinations of input features and target features.

    For all combinations of 2 to (n-1) columns in a predefined list of columns, the function attempts
    to predict each of the remaining columns using a Random Forest model. It uses regression models
    for numeric targets and classification models for categorical targets.

    Args:
        df (pandas.DataFrame): Input DataFrame containing features to be used both as predictors (input) and targets.
        r2_threshold (float): Minimum R² score threshold to consider a regression model significant.
        accuracy_threshold (float): Minimum accuracy threshold to consider a classification model significant.

    Returns:
        list: Currently an empty list (placeholder for future storage of significant model results).

    Note:
        Also prints summary information for models that exceed the specified thresholds.
    """

    # Converts categorical columns to numeric.
    label_encoders = {}
    original_categorical_cols = df.select_dtypes(include='object').columns.tolist()

    for col in original_categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    # Record of the categorical columns after encoding.
    categorical_cols = list(label_encoders.keys())

    # Defines relevant columns to evaluate for input/target relationships.
    all_columns = ['Sample Size', 'GUI Image', 'GUI Model', 'Usability Testing Data',
                   'Comparison Table Data', 'Target System', 'Contribution Focus', 'Year']

    results = []

    # Combinations of 2 to 7 input columns to determine each remaining target column.
    for r in range(2, len(all_columns)):
        input_combinations = list(combinations(all_columns, r))

        for input_cols in input_combinations:
            target_cols = [col for col in all_columns if col not in input_cols]
            X = df[list(input_cols)]

            for target in target_cols:
                y = df[target]

                try:
                    # Uses Regression when target is numeric and Classification when target is categorical.
                    is_regression = target not in categorical_cols

                    stratify_y = y if not is_regression and len(y.unique()) > 1 else None

                    # Split data into training and test sets
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.3, random_state=42, stratify=stratify_y
                    )

                    # Trains the appropriate model.
                    model = RandomForestRegressor(random_state=42) if is_regression else RandomForestClassifier(
                        random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Evaluates model performance.
                    if is_regression:
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)

                        if r2 >= r2_threshold:
                            print("\n===========================")
                            print(f"Input Columns: {input_cols}")
                            print(f"Target Column: {target}")
                            print(f"Feature Importance: {dict(zip(input_cols, model.feature_importances_))}")
                            print(f"Mean Absolute Error: {mae:.4f}")
                            print(f"R² Score: {r2:.4f}")
                    else:
                        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

                        if report["accuracy"] >= accuracy_threshold:
                            print("\n===========================")
                            print(f"Input Columns: {input_cols}")
                            print(f"Target Column: {target}")
                            print(f"Feature Importance: {dict(zip(input_cols, model.feature_importances_))}")
                            print(f"Accuracy: {report['accuracy']:.4f}")

                except Exception as e:
                    print(f"Failed on input {input_cols} and target {target}: {e}")

    return results

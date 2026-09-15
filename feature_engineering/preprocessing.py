"""Fit only on training rows. Raw feature CSVs are never modified."""
def build_preprocessor(level='trajectory', standardize=False):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from . import step_features, trajectory_features
    module = {'step': step_features, 'trajectory': trajectory_features}[level]
    categorical = [c for c in module.FEATURE_COLUMNS if c in {'step_type','agent_id','tool_name','status','dominant_agent'}]
    numerical = [c for c in module.FEATURE_COLUMNS if c not in categorical]
    # Accept a pandas DataFrame read from the CSV; select only declared predictors.
    numeric = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
    if standardize:
        numeric.append(('scale', StandardScaler()))
    return ColumnTransformer([
        ('numeric', Pipeline(numeric), numerical),
        ('categorical', Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='UNKNOWN')),
                                  ('encode',OneHotEncoder(handle_unknown='ignore'))]), categorical),
    ], remainder='drop')

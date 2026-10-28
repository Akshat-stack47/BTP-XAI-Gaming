"""
SHAP and LIME explanation generation for model predictions.
Provides a unified interface for both explanation methods.
"""
import numpy as np
import shap
import lime
import lime.lime_tabular


class ExplanationGenerator:
    """
    Generate feature attributions using SHAP and LIME.
    Provides a consistent interface for both explanation methods.
    """

    def __init__(self, model, X_train, feature_names):
        """
        Parameters
        ----------
        model : DiabetesModel instance
        X_train : training data (for background/reference)
        feature_names : list of feature name strings
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names

        # Initialize SHAP explainer
        if model.model_type == 'rf':
            self.shap_explainer = shap.TreeExplainer(model.model)
        elif model.model_type == 'lr':
            if model.scaler:
                X_bg = model.scaler.transform(X_train[:100])
            else:
                X_bg = X_train[:100]
            self.shap_explainer = shap.LinearExplainer(model.model, X_bg)

        # Initialize LIME explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=['No Diabetes', 'Diabetes'],
            mode='classification',
            random_state=42
        )

    def get_shap_explanation(self, x_single):
        """
        Get SHAP values for a single instance.

        Returns
        -------
        dict: {feature_name: shap_value} sorted by absolute value (descending)
        """
        if self.model.model_type == 'rf':
            sv = self.shap_explainer.shap_values(x_single.reshape(1, -1))
            # For binary classification, use class 1 (diabetes)
            values = sv[1][0] if isinstance(sv, list) else sv[0]
        else:
            if self.model.scaler:
                x_scaled = self.model.scaler.transform(x_single.reshape(1, -1))
            else:
                x_scaled = x_single.reshape(1, -1)
            values = self.shap_explainer.shap_values(x_scaled)[0]

        attributions = dict(zip(self.feature_names, values))
        attributions = dict(sorted(
            attributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))
        return attributions

    def get_lime_explanation(self, x_single, num_features=5):
        """
        Get LIME explanation for a single instance.

        Returns
        -------
        dict: {feature_name: weight} from LIME
        """
        predict_fn = lambda X: np.column_stack([
            1 - self.model.predict_proba(X),
            self.model.predict_proba(X)
        ])

        exp = self.lime_explainer.explain_instance(
            x_single,
            predict_fn,
            num_features=num_features,
            num_samples=500
        )

        return dict(exp.as_list())

    def get_top_gameable_feature(self, attributions, gameable_features):
        """
        Find the most influential feature that the adversary CAN change.

        Parameters
        ----------
        attributions : dict of {feature: importance}
        gameable_features : list of feature names that can be manipulated

        Returns
        -------
        (feature_name, attribution_value) or (None, None)
        """
        for feat, val in attributions.items():
            # LIME returns feature conditions like "feature > X", extract name
            feat_clean = feat.split(' ')[0] if ' ' in feat else feat
            if feat_clean in gameable_features:
                return feat_clean, val
        return None, None

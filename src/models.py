"""
Model definitions and training utilities.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import joblib
import sys
sys.path.append("..")
from config import RANDOM_STATE, MODELS_DIR


def get_models():
    """
    Return a dictionary of model name -> untrained estimator.
    Adjust hyperparameters here or via cross-validation.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            multi_class="multinomial",
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            random_state=RANDOM_STATE,
        ),
    }


def save_model(model, name):
    """Save a trained model to the models/saved directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib"
    joblib.dump(model, path)
    print(f"Saved: {path}")
    return path


def load_model(name):
    """Load a saved model by name."""
    path = MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib"
    return joblib.load(path)

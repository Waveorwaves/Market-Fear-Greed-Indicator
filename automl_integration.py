# automl_integration.py

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
import argparse
warnings.filterwarnings('ignore')

# AutoML libraries (install as needed)
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.svm import SVC, SVR
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from flaml import AutoML
    FLAML_AVAILABLE = True
except ImportError:
    FLAML_AVAILABLE = False

# Local imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Add parent directory to path

class FearGreedAutoML:
    """
    AutoML pipeline for Fear & Greed classification and regression
    Integrates with your feature store and teammate's model approach
    """
    
    def __init__(self, models_dir: str = "models", results_dir: str = "results"):
        # Set up paths relative to current directory (Market-Fear-Greed-Indicator)
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.feature_store_dir = self.project_root / "feature_store"
        
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.classification_models = self._get_classification_models()
        self.regression_models = self._get_regression_models()
        
        # Load feature summary
        self._load_feature_summary()
        
        # Results storage
        self.experiment_results = []
    
    def _load_feature_summary(self):
        """Load feature summary from feature store"""
        summary_file = self.feature_store_dir / "feature_pipeline_summary.json"
        if not summary_file.exists():
            raise FileNotFoundError("Feature summary not found. Run feature store setup first.")
        
        with open(summary_file, 'r') as f:
            self.feature_summary = json.load(f)
        
        print("Local feature summary loaded:")
        for version, info in self.feature_summary.items():
            print(f"  {version}: {info['num_features']} features")
    
    def _get_classification_models(self) -> Dict:
        """Get classification models for sentiment prediction"""
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }
        
        if SKLEARN_AVAILABLE:
            models['svm'] = SVC(random_state=42, probability=True)
        
        return models
    
    def _get_regression_models(self) -> Dict:
        """Get regression models for VIX prediction"""
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0, random_state=42),
            'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }
        
        if SKLEARN_AVAILABLE:
            models['svr'] = SVR()
        
        return models
    
    def prepare_data(self, version: str = 'v3', task_type: str = 'classification',
                    test_size: float = 0.2, validation_size: float = 0.2) -> Dict:
        """
        Prepare data for AutoML training using local processed files
        Following your teammate's chronological split approach
        """
        print(f"Preparing data for {task_type} task (version: {version})...")
        
        if version not in self.feature_summary:
            raise ValueError(f"Version {version} not found. Available: {list(self.feature_summary.keys())}")
        
        # Load data from local parquet file
        version_info = self.feature_summary[version]
        local_name = version_info['local_name']
        feature_columns = version_info['features']
        
        data_file = self.feature_store_dir / f"{local_name}.parquet"
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        df = pd.read_parquet(data_file)
        print(f"Loaded dataset: {df.shape}")
        
        # Prepare features and target
        X = df[feature_columns].copy()
        
        if task_type == 'classification':
            y = df['Market_Sentiment'].copy()
            # Remove NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X, y = X[mask], y[mask]
            
            # Encode labels
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            target_info = {'encoder': le, 'classes': le.classes_}
            
        else:  # regression
            y = df['VIX_Target'].copy()
            # Remove NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X, y = X[mask], y[mask]
            y_encoded = y.values
            target_info = {'mean': y.mean(), 'std': y.std()}
        
        # Chronological split (following your teammate's approach)
        # Use the last portion as test set
        n_samples = len(X)
        test_start_idx = int(n_samples * (1 - test_size))
        
        X_temp, X_test = X.iloc[:test_start_idx], X.iloc[test_start_idx:]
        y_temp, y_test = y_encoded[:test_start_idx], y_encoded[test_start_idx:]
        
        # Split remaining data into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size/(1-test_size), 
            random_state=42, stratify=y_temp if task_type == 'classification' else None
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        data_dict = {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': feature_columns,
            'scaler': scaler,
            'target_info': target_info,
            'task_type': task_type,
            'version': version
        }
        
        print(f"Data prepared:")
        print(f"   - Train: {X_train_scaled.shape}")
        print(f"   - Validation: {X_val_scaled.shape}")
        print(f"   - Test: {X_test_scaled.shape}")
        print(f"   - Features: {len(feature_columns)}")
        
        return data_dict
    
    def run_sklearn_automl(self, data_dict: Dict, time_budget: int = 300) -> Dict:
        """
        Run AutoML using sklearn models with basic hyperparameter search
        """
        task_type = data_dict['task_type']
        models = self.classification_models if task_type == 'classification' else self.regression_models
        
        print(f"Running sklearn AutoML for {task_type}...")
        print(f"   - Models: {list(models.keys())}")
        print(f"   - Time budget: {time_budget}s")
        
        results = []
        best_score = -np.inf if task_type == 'classification' else np.inf
        best_model = None
        best_model_name = None
        
        X_train, y_train = data_dict['X_train'], data_dict['y_train']
        X_val, y_val = data_dict['X_val'], data_dict['y_val']
        
        for model_name, model in models.items():
            print(f"   Training {model_name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred_train = model.predict(X_train)
                y_pred_val = model.predict(X_val)
                
                if task_type == 'classification':
                    train_score = accuracy_score(y_train, y_pred_train)
                    val_score = accuracy_score(y_val, y_pred_val)
                    
                    # Get classification report
                    target_names = data_dict['target_info']['classes']
                    report = classification_report(y_val, y_pred_val, 
                                                 target_names=target_names, 
                                                 output_dict=True, zero_division=0)
                    
                    result = {
                        'model_name': model_name,
                        'train_accuracy': train_score,
                        'val_accuracy': val_score,
                        'precision': report['macro avg']['precision'],
                        'recall': report['macro avg']['recall'],
                        'f1_score': report['macro avg']['f1-score']
                    }
                    
                    # Update best model (higher accuracy is better)
                    if val_score > best_score:
                        best_score = val_score
                        best_model = model
                        best_model_name = model_name
                
                else:  # regression
                    train_score = r2_score(y_train, y_pred_train)
                    val_score = r2_score(y_val, y_pred_val)
                    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
                    
                    result = {
                        'model_name': model_name,
                        'train_r2': train_score,
                        'val_r2': val_score,
                        'train_rmse': train_rmse,
                        'val_rmse': val_rmse
                    }
                    
                    # Update best model (lower RMSE is better)
                    if val_rmse < best_score or best_score == np.inf:
                        best_score = val_rmse
                        best_model = model
                        best_model_name = model_name
                
                results.append(result)
                print(f"     SUCCESS: {model_name}")
                
            except Exception as e:
                print(f"     ERROR: {model_name} failed: {str(e)}")
                continue
        
        return {
            'results': results,
            'best_model': best_model,
            'best_model_name': best_model_name,
            'best_score': best_score
        }
    
    def run_flaml_automl(self, data_dict: Dict, time_budget: int = 300) -> Optional[Dict]:
        """
        Run AutoML using FLAML (if available)
        """
        if not FLAML_AVAILABLE:
            print("WARNING: FLAML not available. Install with: pip install flaml")
            return None
        
        task_type = data_dict['task_type']
        print(f"Running FLAML AutoML for {task_type}...")
        
        try:
            automl = AutoML()
            
            # Configure AutoML
            settings = {
                'time_budget': time_budget,
                'metric': 'accuracy' if task_type == 'classification' else 'r2',
                'task': task_type,
                'log_file_name': str(self.results_dir / f'flaml_{task_type}.log'),
                'seed': 42,
                'estimator_list': ['lgbm', 'xgboost', 'rf']
            }
            
            # Train
            automl.fit(data_dict['X_train'], data_dict['y_train'], **settings)
            
            # Predict
            y_pred_val = automl.predict(data_dict['X_val'])
            
            if task_type == 'classification':
                val_score = accuracy_score(data_dict['y_val'], y_pred_val)
                metric_name = 'accuracy'
            else:
                val_score = r2_score(data_dict['y_val'], y_pred_val)
                metric_name = 'r2'
            
            print(f"FLAML completed - Best {metric_name}: {val_score:.4f}")
            print(f"Best estimator: {automl.best_estimator}")
            
            return {
                'automl_model': automl,
                'best_estimator': automl.best_estimator,
                'best_config': automl.best_config,
                'val_score': val_score
            }
            
        except Exception as e:
            print(f"ERROR: FLAML failed: {str(e)}")
            return None
    
    def evaluate_on_test_set(self, model, data_dict: Dict) -> Dict:
        """
        Evaluate the best model on test set
        """
        task_type = data_dict['task_type']
        X_test, y_test = data_dict['X_test'], data_dict['y_test']
        
        print(f"Evaluating on test set...")
        
        # Predict
        y_pred_test = model.predict(X_test)
        
        if task_type == 'classification':
            test_accuracy = accuracy_score(y_test, y_pred_test)
            target_names = data_dict['target_info']['classes']
            report = classification_report(y_test, y_pred_test, 
                                         target_names=target_names, 
                                         output_dict=True, zero_division=0)
            
            results = {
                'test_accuracy': test_accuracy,
                'precision': report['macro avg']['precision'],
                'recall': report['macro avg']['recall'],
                'f1_score': report['macro avg']['f1-score'],
                'classification_report': report
            }
            
            print(f"   Test Accuracy: {test_accuracy:.4f}")
            print(f"   F1-Score: {report['macro avg']['f1-score']:.4f}")
            
        else:  # regression
            test_r2 = r2_score(y_test, y_pred_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            results = {
                'test_r2': test_r2,
                'test_rmse': test_rmse
            }
            
            print(f"   Test R²: {test_r2:.4f}")
            print(f"   Test RMSE: {test_rmse:.4f}")
        
        return results
    
    def save_experiment_results(self, experiment_name: str, results: Dict) -> str:
        """
        Save experiment results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"{experiment_name}_{timestamp}.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean results for JSON serialization
        clean_results = {}
        for key, value in results.items():
            if key in ['best_model', 'automl_model', 'scaler', 'target_info']:
                continue  # Skip non-serializable objects
            clean_results[key] = convert_numpy(value)
        
        with open(results_file, 'w') as f:
            json.dump(clean_results, f, indent=2, default=str)
        
        print(f"Results saved: {results_file}")
        return str(results_file)
    
    def run_full_automl_pipeline(self, version: str = 'v3', 
                                task_type: str = 'classification',
                                time_budget: int = 300) -> Dict:
        """
        Run the complete AutoML pipeline
        """
        print(f"Running Full AutoML Pipeline")
        print(f"   Version: {version}")
        print(f"   Task: {task_type}")
        print("=" * 60)
        
        # Prepare data
        data_dict = self.prepare_data(version, task_type)
        
        # Run sklearn AutoML
        sklearn_results = self.run_sklearn_automl(data_dict, time_budget)
        
        # Run FLAML AutoML (if available)
        flaml_results = self.run_flaml_automl(data_dict, time_budget)
        
        # Determine best model
        best_model = sklearn_results['best_model']
        best_source = 'sklearn'
        
        if flaml_results and flaml_results['val_score'] > sklearn_results['best_score']:
            best_model = flaml_results['automl_model']
            best_source = 'flaml'
        
        # Evaluate on test set
        test_results = self.evaluate_on_test_set(best_model, data_dict)
        
        # Compile final results
        final_results = {
            'experiment_name': f"{version}_{task_type}",
            'version': version,
            'task_type': task_type,
            'best_source': best_source,
            'sklearn_results': sklearn_results['results'],
            'flaml_available': FLAML_AVAILABLE,
            'flaml_results': flaml_results,
            'test_results': test_results,
            'data_info': {
                'n_features': len(data_dict['feature_names']),
                'train_samples': data_dict['X_train'].shape[0],
                'val_samples': data_dict['X_val'].shape[0],
                'test_samples': data_dict['X_test'].shape[0]
            }
        }
        
        # Save results
        results_file = self.save_experiment_results(
            f"{version}_{task_type}", final_results
        )
        
        # Save best model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = self.models_dir / f"best_{version}_{task_type}_{timestamp}.joblib"
        joblib.dump({
            'model': best_model,
            'scaler': data_dict['scaler'],
            'feature_names': data_dict['feature_names'],
            'target_info': data_dict['target_info'],
            'version': version,
            'task_type': task_type,
            'best_source': best_source
        }, model_file)
        
        print(f"\nAutoML Pipeline Complete!")
        print(f"   Best Model: {best_source}")
        print(f"   Model saved: {model_file}")
        print(f"   Results saved: {results_file}")
        
        return final_results

def main():
    """
    Main function to run AutoML experiments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='v3', help='Feature version (v1, v2, v3)')
    parser.add_argument('--task', type=str, default='both', help='Task type (classification, regression, both)')
    parser.add_argument('--time_budget', type=int, default=300, help='Time budget for FLAML (seconds)')
    parser.add_argument('--models_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')
    args = parser.parse_args()

    print("Local Fear & Greed AutoML Pipeline")
    print("=" * 60)
    
    # Initialize AutoML
    automl = FearGreedAutoML(
        models_dir=args.models_dir,
        results_dir=args.results_dir
    )
    
    # Run experiments
    tasks = ['classification', 'regression'] if args.task == 'both' else [args.task]
    
    all_results = {}
    for task in tasks:
        print(f"\n### Running {task} for {args.version} ###")
        try:
            results = automl.run_full_automl_pipeline(
                version=args.version,
                task_type=task,
                time_budget=args.time_budget
            )
            all_results[task] = results
        except Exception as e:
            print(f"ERROR: {task} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n### SUMMARY ###")
    for task, results in all_results.items():
        print(f"{task.upper()}:")
        print(f"  Best model: {results['best_source']}")
        print(f"  Features: {results['data_info']['n_features']}")

if __name__ == "__main__":
    main()

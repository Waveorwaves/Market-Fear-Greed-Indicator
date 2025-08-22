#!/usr/bin/env python3
"""
Evidently-based Monitoring System for Fear & Greed Classifier
Comprehensive data drift detection and model performance monitoring
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import warnings
import sys
import joblib

# Evidently imports (updated for newer version)
try:
    from evidently.pipeline.column_mapping import ColumnMapping
except ImportError:
    from evidently import ColumnMapping

from evidently.report import Report
from evidently.metrics import *
from evidently.test_suite import TestSuite
from evidently.tests import *

# Workspace
try:
    from evidently.ui.workspace import Workspace
except ImportError:
    print("⚠️ Workspace functionality not available in this Evidently version")

warnings.filterwarnings('ignore')

class FearGreedEvidentlyMonitor:
    """
    Evidently-based monitoring system for Fear & Greed classifier with comprehensive
    data drift detection, model performance tracking, and interactive dashboards.
    """
    
    def __init__(self, project_root: str = "."):
        """
        Initialize Evidently monitoring system
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.results_dir = self.project_root / "results"
        self.monitoring_dir = self.project_root / "monitoring"
        
        # Create monitoring directory
        self.monitoring_dir.mkdir(exist_ok=True)
        
        print(f"✅ Evidently Monitor initialized")
        print(f"   Project root: {self.project_root.resolve()}")
        print(f"   Monitoring dir: {self.monitoring_dir.resolve()}")
    
    def load_holdout_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare old and new holdout datasets"""
        old_file = self.data_dir / "old_future_holdout_20pct.csv"
        new_file = self.data_dir / "new_future_holdout_20pct.csv"
        
        if not old_file.exists():
            raise FileNotFoundError(f"Old holdout file not found: {old_file}")
        if not new_file.exists():
            raise FileNotFoundError(f"New holdout file not found: {new_file}")
        
        old_data = pd.read_csv(old_file)
        new_data = pd.read_csv(new_file)
        
        # Convert Date columns
        old_data['Date'] = pd.to_datetime(old_data['Date'])
        new_data['Date'] = pd.to_datetime(new_data['Date'])
        
        # Sort by date
        old_data = old_data.sort_values('Date').reset_index(drop=True)
        new_data = new_data.sort_values('Date').reset_index(drop=True)
        
        print(f"📊 Loaded holdout data:")
        print(f"   Old holdout: {old_data.shape[0]} samples")
        print(f"   New holdout: {new_data.shape[0]} samples")
        print(f"   Date range old: {old_data['Date'].min()} to {old_data['Date'].max()}")
        print(f"   Date range new: {new_data['Date'].min()} to {new_data['Date'].max()}")
        
        return old_data, new_data
    
    def setup_column_mapping(self, data: pd.DataFrame) -> ColumnMapping:
        """Setup column mapping for Evidently"""
        
        # Define feature categories
        numerical_features = [
            'SPY_Price', 'SPY_Volatility', 'SPY_Volume',
            'QQQ_Price', 'QQQ_Volatility', 'QQQ_Volume', 
            'GLD_Price', 'GLD_Volatility', 'GLD_Volume',
            'HYG_Price', 'HYG_Volatility', 'HYG_Volume',
            'VIX_Price', 'TLT_Price', 'TLT_Volume', 'TLT_Volatility',
            'CPIAUCSL', 'HOUST', 'BAMLH0A0HYM2', 'DGS10', 'UMCSENT'
        ]
        
        # Filter to available columns
        available_numerical = [col for col in numerical_features if col in data.columns]
        
        categorical_features = ['VIX_Fear_Level'] if 'VIX_Fear_Level' in data.columns else []
        
        column_mapping = ColumnMapping(
            target='Market_Sentiment' if 'Market_Sentiment' in data.columns else None,
            prediction=None,  # We'll add this when evaluating models
            numerical_features=available_numerical,
            categorical_features=categorical_features,
            datetime_features=['Date'] if 'Date' in data.columns else None
        )
        
        return column_mapping
    
    def create_data_drift_report(self, 
                               reference_data: pd.DataFrame, 
                               current_data: pd.DataFrame) -> Report:
        """Create comprehensive data drift report"""
        
        column_mapping = self.setup_column_mapping(reference_data)
        
        # Create data drift report
        data_drift_report = Report(metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            DataDriftTable(),
            ColumnDriftMetric(column_name='SPY_Price'),
            ColumnDriftMetric(column_name='VIX_Price'),
            ColumnDriftMetric(column_name='QQQ_Price'),
            ColumnDriftMetric(column_name='DGS10'),
            ColumnSummaryMetric(column_name='Market_Sentiment'),
            ColumnDistributionMetric(column_name='SPY_Volatility'),
            ColumnDistributionMetric(column_name='VIX_Price'),
            ColumnCorrelationsMetric(),
            DatasetCorrelationsMetric(),
        ])
        
        print("🔍 Generating data drift report...")
        data_drift_report.run(
            reference_data=reference_data,
            current_data=current_data, 
            column_mapping=column_mapping
        )
        
        return data_drift_report
    
    def create_data_quality_report(self, 
                                 reference_data: pd.DataFrame,
                                 current_data: pd.DataFrame) -> Report:
        """Create data quality monitoring report"""
        
        column_mapping = self.setup_column_mapping(reference_data)
        
        data_quality_report = Report(metrics=[
            DatasetMissingValuesMetric(),
            DataQualityMetric(),
            ColumnMissingValuesMetric(column_name='SPY_Price'),
            ColumnMissingValuesMetric(column_name='VIX_Price'),
            ColumnRegExpMetric(column_name='Market_Sentiment', reg_exp=r'^(stable|fear|greed)$'),
            DatasetSummaryMetric(),
            ColumnQuantileMetric(column_name='SPY_Price', quantile=0.05),
            ColumnQuantileMetric(column_name='SPY_Price', quantile=0.95),
            ColumnQuantileMetric(column_name='VIX_Price', quantile=0.05),
            ColumnQuantileMetric(column_name='VIX_Price', quantile=0.95),
        ])
        
        print("📋 Generating data quality report...")
        data_quality_report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        return data_quality_report
    
    def evaluate_models_with_evidently(self, 
                                     reference_data: pd.DataFrame,
                                     current_data: pd.DataFrame) -> Dict[str, Report]:
        """Evaluate models and create model performance reports"""
        
        model_reports = {}
        
        # Load feature store for feature engineering
        try:
            sys.path.append(str(self.project_root))
            from feature_store_setup import FearGreedFeatureStore
            
            fs = FearGreedFeatureStore(
                data_dir=str(self.data_dir),
                feature_store_dir=str(self.project_root / "feature_store")
            )
            
            # Process both datasets
            ref_with_targets = fs.create_target_variable(reference_data.copy())
            curr_with_targets = fs.create_target_variable(current_data.copy())
            
            # Test each version
            for version in ['v1', 'v2', 'v3']:
                print(f"🤖 Evaluating {version} models...")
                
                try:
                    # Create features
                    if version == 'v1':
                        ref_features, feature_names = fs.implement_v1_features(ref_with_targets)
                        curr_features, _ = fs.implement_v1_features(curr_with_targets)
                    elif version == 'v2':
                        ref_features, feature_names = fs.implement_v2_features(ref_with_targets)
                        curr_features, _ = fs.implement_v2_features(curr_with_targets)
                    elif version == 'v3':
                        ref_features, feature_names = fs.implement_v3_features(ref_with_targets)
                        curr_features, _ = fs.implement_v3_features(curr_with_targets)
                    
                    # Clean data
                    ref_features = ref_features.dropna()
                    curr_features = curr_features.dropna()
                    
                    if len(ref_features) == 0 or len(curr_features) == 0:
                        print(f"   ⚠️ No valid samples for {version}")
                        continue
                    
                    # Find classification model
                    class_model_files = list(self.models_dir.glob(f'*{version}_classification*.joblib'))
                    if class_model_files:
                        latest_class_model = max(class_model_files, key=lambda x: x.stat().st_mtime)
                        
                        try:
                            model_data = joblib.load(latest_class_model)
                            model = model_data['model']
                            
                            # Make predictions
                            X_ref = ref_features[feature_names]
                            X_curr = curr_features[feature_names]
                            y_ref = ref_features['Market_Sentiment']
                            y_curr = curr_features['Market_Sentiment']
                            
                            ref_predictions = model.predict(X_ref)
                            curr_predictions = model.predict(X_curr)
                            
                            # Add predictions to dataframes
                            ref_with_pred = ref_features.copy()
                            ref_with_pred['prediction'] = ref_predictions
                            
                            curr_with_pred = curr_features.copy()
                            curr_with_pred['prediction'] = curr_predictions
                            
                            # Create column mapping for classification
                            class_column_mapping = ColumnMapping(
                                target='Market_Sentiment',
                                prediction='prediction',
                                numerical_features=feature_names[:10],  # Subset for performance
                                task='classification'
                            )
                            
                            # Create classification performance report
                            classification_report = Report(metrics=[
                                ClassificationQualityMetric(),
                                ClassificationClassBalance(),
                                ClassificationConfusionMatrix(),
                                ClassificationQualityByClass(),
                                TargetDriftMetric(),
                                PredictionDriftMetric(),
                                ClassificationDummyMetric(),
                            ])
                            
                            classification_report.run(
                                reference_data=ref_with_pred,
                                current_data=curr_with_pred,
                                column_mapping=class_column_mapping
                            )
                            
                            model_reports[f'{version}_classification'] = classification_report
                            print(f"   ✅ {version} classification report created")
                            
                        except Exception as e:
                            print(f"   ❌ Error with {version} classification: {e}")
                    
                    # Find regression model
                    reg_model_files = list(self.models_dir.glob(f'*{version}_regression*.joblib'))
                    if reg_model_files and 'VIX_Target' in ref_features.columns:
                        latest_reg_model = max(reg_model_files, key=lambda x: x.stat().st_mtime)
                        
                        try:
                            model_data = joblib.load(latest_reg_model)
                            model = model_data['model']
                            
                            # Make predictions
                            X_ref = ref_features[feature_names]
                            X_curr = curr_features[feature_names]
                            
                            ref_predictions = model.predict(X_ref)
                            curr_predictions = model.predict(X_curr)
                            
                            # Add predictions to dataframes
                            ref_with_pred = ref_features.copy()
                            ref_with_pred['prediction'] = ref_predictions
                            
                            curr_with_pred = curr_features.copy()
                            curr_with_pred['prediction'] = curr_predictions
                            
                            # Create column mapping for regression
                            reg_column_mapping = ColumnMapping(
                                target='VIX_Target',
                                prediction='prediction',
                                numerical_features=feature_names[:10],  # Subset for performance
                                task='regression'
                            )
                            
                            # Create regression performance report
                            regression_report = Report(metrics=[
                                RegressionQualityMetric(),
                                RegressionPredictedVsActualScatter(),
                                RegressionPredictedVsActualPlot(),
                                RegressionErrorPlot(),
                                RegressionAbsPercentageErrorPlot(),
                                TargetDriftMetric(),
                                PredictionDriftMetric(),
                                RegressionDummyMetric(),
                            ])
                            
                            regression_report.run(
                                reference_data=ref_with_pred,
                                current_data=curr_with_pred,
                                column_mapping=reg_column_mapping
                            )
                            
                            model_reports[f'{version}_regression'] = regression_report
                            print(f"   ✅ {version} regression report created")
                            
                        except Exception as e:
                            print(f"   ❌ Error with {version} regression: {e}")
                
                except Exception as e:
                    print(f"   ❌ Error processing {version}: {e}")
        
        except ImportError as e:
            print(f"⚠️ Could not import feature store: {e}")
        
        return model_reports
    
    def create_test_suite(self, 
                         reference_data: pd.DataFrame,
                         current_data: pd.DataFrame) -> TestSuite:
        """Create test suite for automated monitoring"""
        
        column_mapping = self.setup_column_mapping(reference_data)
        
        test_suite = TestSuite(tests=[
            TestNumberOfColumnsWithMissingValues(),
            TestNumberOfRowsWithMissingValues(),
            TestNumberOfConstantColumns(),
            TestNumberOfDriftedColumns(),
            TestShareOfMissingValues(),
            TestColumnDrift(column_name='SPY_Price'),
            TestColumnDrift(column_name='VIX_Price'),
            TestColumnDrift(column_name='Market_Sentiment'),
            TestMeanInNSigmas(column_name='SPY_Price'),
            TestValueRange(column_name='VIX_Price', left=5, right=80),
            TestNumberOfUniqueValues(column_name='Market_Sentiment', eq=3),
        ])
        
        print("🧪 Running test suite...")
        test_suite.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        return test_suite
    
    def save_reports(self, 
                    data_drift_report: Report,
                    data_quality_report: Report, 
                    model_reports: Dict[str, Report],
                    test_suite: TestSuite) -> Dict[str, str]:
        """Save all reports as HTML files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        # Save data drift report
        drift_file = self.monitoring_dir / f"data_drift_report_{timestamp}.html"
        data_drift_report.save_html(str(drift_file))
        saved_files['data_drift'] = str(drift_file)
        print(f"💾 Data drift report saved: {drift_file}")
        
        # Save data quality report
        quality_file = self.monitoring_dir / f"data_quality_report_{timestamp}.html"
        data_quality_report.save_html(str(quality_file))
        saved_files['data_quality'] = str(quality_file)
        print(f"💾 Data quality report saved: {quality_file}")
        
        # Save model performance reports
        for model_name, report in model_reports.items():
            model_file = self.monitoring_dir / f"model_performance_{model_name}_{timestamp}.html"
            report.save_html(str(model_file))
            saved_files[f'model_{model_name}'] = str(model_file)
            print(f"💾 Model report saved: {model_file}")
        
        # Save test suite
        test_file = self.monitoring_dir / f"test_suite_{timestamp}.html"
        test_suite.save_html(str(test_file))
        saved_files['test_suite'] = str(test_file)
        print(f"💾 Test suite saved: {test_file}")
        
        return saved_files
    
    def create_monitoring_summary(self, 
                                data_drift_report: Report,
                                model_reports: Dict[str, Report],
                                test_suite: TestSuite) -> Dict:
        """Create monitoring summary with key metrics"""
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_drift': {},
            'model_performance': {},
            'test_results': {},
            'alerts': []
        }
        
        # Extract data drift metrics
        try:
            drift_results = data_drift_report.as_dict()
            if 'metrics' in drift_results:
                for metric in drift_results['metrics']:
                    if metric.get('metric') == 'DatasetDriftMetric':
                        result = metric.get('result', {})
                        summary['data_drift'] = {
                            'drift_share': result.get('drift_share', 0),
                            'number_of_drifted_columns': result.get('number_of_drifted_columns', 0),
                            'dataset_drift': result.get('dataset_drift', False)
                        }
                        
                        if result.get('dataset_drift', False):
                            summary['alerts'].append("🚨 Dataset drift detected!")
        except Exception as e:
            print(f"⚠️ Error extracting drift metrics: {e}")
        
        # Extract model performance metrics
        for model_name, report in model_reports.items():
            try:
                model_results = report.as_dict()
                if 'metrics' in model_results:
                    for metric in model_results['metrics']:
                        if metric.get('metric') == 'ClassificationQualityMetric':
                            result = metric.get('result', {})
                            current_metrics = result.get('current', {})
                            summary['model_performance'][model_name] = {
                                'accuracy': current_metrics.get('accuracy'),
                                'precision': current_metrics.get('precision'),
                                'recall': current_metrics.get('recall'),
                                'f1': current_metrics.get('f1')
                            }
                        elif metric.get('metric') == 'RegressionQualityMetric':
                            result = metric.get('result', {})
                            current_metrics = result.get('current', {})
                            summary['model_performance'][model_name] = {
                                'mae': current_metrics.get('mean_abs_error'),
                                'mape': current_metrics.get('mean_abs_perc_error'),
                                'r2': current_metrics.get('r2_score')
                            }
            except Exception as e:
                print(f"⚠️ Error extracting metrics for {model_name}: {e}")
        
        # Extract test results
        try:
            test_results = test_suite.as_dict()
            if 'tests' in test_results:
                total_tests = len(test_results['tests'])
                passed_tests = sum(1 for test in test_results['tests'] if test.get('status') == 'SUCCESS')
                summary['test_results'] = {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'success_rate': passed_tests / total_tests if total_tests > 0 else 0
                }
                
                if passed_tests < total_tests:
                    failed_count = total_tests - passed_tests
                    summary['alerts'].append(f"⚠️ {failed_count} tests failed!")
        except Exception as e:
            print(f"⚠️ Error extracting test results: {e}")
        
        return summary
    
    def setup_workspace(self) -> str:
        """Setup Evidently workspace for ongoing monitoring"""
        
        try:
            workspace_path = self.monitoring_dir / "workspace"
            workspace = Workspace.create(str(workspace_path))
            
            # Create project
            project = workspace.create_project("Fear_Greed_Monitoring")
            project.description = "Continuous monitoring of Fear & Greed classifier data drift and model performance"
            
            print(f"📊 Evidently workspace created: {workspace_path}")
            print(f"🎯 Project: {project.name}")
            
            return str(workspace_path)
        except Exception as e:
            print(f"⚠️ Workspace setup not available: {e}")
            return str(self.monitoring_dir)
    
    def run_complete_monitoring(self) -> Dict:
        """Run complete monitoring pipeline"""
        
        print("🚀 Starting Complete Evidently Monitoring Pipeline")
        print("=" * 60)
        
        # Load data
        reference_data, current_data = self.load_holdout_data()
        
        # Create data drift report
        print("\n📊 Creating data drift report...")
        data_drift_report = self.create_data_drift_report(reference_data, current_data)
        
        # Create data quality report
        print("\n📋 Creating data quality report...")
        data_quality_report = self.create_data_quality_report(reference_data, current_data)
        
        # Evaluate models
        print("\n🤖 Evaluating model performance...")
        model_reports = self.evaluate_models_with_evidently(reference_data, current_data)
        
        # Create test suite
        print("\n🧪 Running test suite...")
        test_suite = self.create_test_suite(reference_data, current_data)
        
        # Save all reports
        print("\n💾 Saving reports...")
        saved_files = self.save_reports(data_drift_report, data_quality_report, model_reports, test_suite)
        
        # Create summary
        print("\n📝 Creating monitoring summary...")
        summary = self.create_monitoring_summary(data_drift_report, model_reports, test_suite)
        
        # Save summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.monitoring_dir / f"monitoring_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        summary['saved_files'] = saved_files
        summary['summary_file'] = str(summary_file)
        
        print(f"\n✅ Complete monitoring pipeline finished!")
        print(f"📄 Summary saved: {summary_file}")
        print(f"📁 All reports in: {self.monitoring_dir}")
        
        # Print key findings
        if summary.get('data_drift', {}).get('dataset_drift'):
            print("🚨 ALERT: Dataset drift detected!")
        else:
            print("✅ No significant dataset drift detected")
        
        if summary.get('alerts'):
            print("⚠️ Alerts:")
            for alert in summary['alerts']:
                print(f"   {alert}")
        
        return summary


def main():
    """Main function to run monitoring"""
    monitor = FearGreedEvidentlyMonitor()
    results = monitor.run_complete_monitoring()
    
    print(f"\n🎉 Monitoring completed successfully!")
    print(f"📊 Open any HTML report to view detailed results")
    
    return results


if __name__ == "__main__":
    results = main()
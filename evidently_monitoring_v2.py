#!/usr/bin/env python3
"""
Evidently-based Monitoring System for Fear & Greed Classifier (v0.7.11 compatible)
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

# Basic imports for monitoring
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

# Evidently imports for version 0.7.11
try:
    from evidently import Report
    from evidently.reports import DataDriftReport, ModelPerformanceReport, ModelQualityReport
    print("✅ Evidently 0.7.11 imports successful")
    EVIDENTLY_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Evidently import failed: {e}")
    EVIDENTLY_AVAILABLE = False

warnings.filterwarnings('ignore')

class FearGreedEvidentlyMonitorV2:
    """
    Evidently-based monitoring system compatible with version 0.7.11
    Focuses on data drift detection and model performance tracking
    """
    
    def __init__(self, project_root: str = "."):
        """Initialize monitoring system"""
        self.project_root = Path(project_root)
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.results_dir = self.project_root / "results"
        self.monitoring_dir = self.project_root / "monitoring"
        
        # Create monitoring directory
        self.monitoring_dir.mkdir(exist_ok=True)
        
        print(f"✅ Evidently Monitor V2 initialized")
        print(f"   Project root: {self.project_root.resolve()}")
        print(f"   Monitoring dir: {self.monitoring_dir.resolve()}")
        print(f"   Evidently available: {EVIDENTLY_AVAILABLE}")
    
    def load_holdout_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare old and new holdout datasets"""
        old_file = self.data_dir / "old_future_holdout_20pct.csv"
        new_file = self.data_dir / "new_future_holdout_20pct.csv"
        
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
    
    def manual_drift_detection(self, 
                             reference_data: pd.DataFrame, 
                             current_data: pd.DataFrame) -> Dict:
        """Manual drift detection using statistical tests"""
        
        numerical_features = [
            'SPY_Price', 'SPY_Volatility', 'SPY_Volume',
            'QQQ_Price', 'QQQ_Volatility', 'QQQ_Volume',
            'GLD_Price', 'GLD_Volatility', 'GLD_Volume',
            'HYG_Price', 'HYG_Volatility', 'HYG_Volume',
            'VIX_Price', 'TLT_Price', 'TLT_Volume', 'TLT_Volatility',
            'CPIAUCSL', 'HOUST', 'BAMLH0A0HYM2', 'DGS10', 'UMCSENT'
        ]
        
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'features_tested': 0,
            'features_with_drift': 0,
            'drift_details': {},
            'target_drift': {},
            'summary': {}
        }
        
        # Test numerical features
        for feature in numerical_features:
            if feature in reference_data.columns and feature in current_data.columns:
                ref_values = reference_data[feature].dropna()
                curr_values = current_data[feature].dropna()
                
                if len(ref_values) > 10 and len(curr_values) > 10:
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_pvalue = stats.ks_2samp(ref_values, curr_values)
                    
                    # Mann-Whitney U test (non-parametric)
                    mw_stat, mw_pvalue = stats.mannwhitneyu(ref_values, curr_values, alternative='two-sided')
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(ref_values) - 1) * ref_values.var() + 
                                         (len(curr_values) - 1) * curr_values.var()) / 
                                        (len(ref_values) + len(curr_values) - 2))
                    cohens_d = (curr_values.mean() - ref_values.mean()) / pooled_std if pooled_std > 0 else 0
                    
                    drift_detected = ks_pvalue < 0.05 or mw_pvalue < 0.05
                    
                    drift_results['drift_details'][feature] = {
                        'ks_statistic': float(ks_stat),
                        'ks_pvalue': float(ks_pvalue),
                        'mw_pvalue': float(mw_pvalue),
                        'cohens_d': float(cohens_d),
                        'drift_detected': drift_detected,
                        'ref_mean': float(ref_values.mean()),
                        'curr_mean': float(curr_values.mean()),
                        'ref_std': float(ref_values.std()),
                        'curr_std': float(curr_values.std()),
                        'mean_shift_pct': float((curr_values.mean() - ref_values.mean()) / ref_values.mean() * 100)
                    }
                    
                    drift_results['features_tested'] += 1
                    if drift_detected:
                        drift_results['features_with_drift'] += 1
        
        # Test target distribution
        if 'Market_Sentiment' in reference_data.columns and 'Market_Sentiment' in current_data.columns:
            ref_target = reference_data['Market_Sentiment'].value_counts()
            curr_target = current_data['Market_Sentiment'].value_counts()
            
            # Chi-square test for categorical distribution
            all_categories = set(ref_target.index) | set(curr_target.index)
            ref_counts = [ref_target.get(cat, 0) for cat in all_categories]
            curr_counts = [curr_target.get(cat, 0) for cat in all_categories]
            
            if sum(ref_counts) > 0 and sum(curr_counts) > 0:
                chi2_stat, chi2_pvalue, _, _ = stats.chi2_contingency([ref_counts, curr_counts])
                
                drift_results['target_drift'] = {
                    'chi2_statistic': float(chi2_stat),
                    'chi2_pvalue': float(chi2_pvalue),
                    'drift_detected': chi2_pvalue < 0.05,
                    'ref_distribution': ref_target.to_dict(),
                    'curr_distribution': curr_target.to_dict()
                }
        
        # Summary
        drift_results['summary'] = {
            'drift_percentage': (drift_results['features_with_drift'] / drift_results['features_tested'] * 100) if drift_results['features_tested'] > 0 else 0,
            'overall_drift_detected': drift_results['features_with_drift'] > 0 or drift_results.get('target_drift', {}).get('drift_detected', False)
        }
        
        return drift_results
    
    def create_evidently_reports(self, 
                               reference_data: pd.DataFrame,
                               current_data: pd.DataFrame) -> Dict:
        """Create Evidently reports if available"""
        
        if not EVIDENTLY_AVAILABLE:
            return {}
        
        reports = {}
        
        try:
            # Try to create data drift report
            data_drift_report = DataDriftReport()
            data_drift_report.run(reference_data=reference_data, current_data=current_data)
            reports['data_drift'] = data_drift_report
            print("✅ Data drift report created")
            
        except Exception as e:
            print(f"⚠️ Data drift report failed: {e}")
        
        try:
            # If we have target predictions, create model performance report
            if 'prediction' in current_data.columns:
                model_performance_report = ModelPerformanceReport()
                model_performance_report.run(reference_data=reference_data, current_data=current_data)
                reports['model_performance'] = model_performance_report
                print("✅ Model performance report created")
        
        except Exception as e:
            print(f"⚠️ Model performance report failed: {e}")
        
        return reports
    
    def evaluate_models(self, 
                       reference_data: pd.DataFrame,
                       current_data: pd.DataFrame) -> Dict:
        """Evaluate models on new data"""
        
        evaluation_results = {
            'timestamp': datetime.now().isoformat(),
            'model_performance': {},
            'model_comparison': {}
        }
        
        # Load feature store
        try:
            sys.path.append(str(self.project_root))
            from feature_store_setup import FearGreedFeatureStore
            
            fs = FearGreedFeatureStore(
                data_dir=str(self.data_dir),
                feature_store_dir=str(self.project_root / "feature_store")
            )
            
            # Process data
            ref_with_targets = fs.create_target_variable(reference_data.copy())
            curr_with_targets = fs.create_target_variable(current_data.copy())
            
            # Evaluate each version
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
                    
                    if len(curr_features) == 0:
                        continue
                    
                    # Evaluate classification models
                    class_model_files = list(self.models_dir.glob(f'*{version}_classification*.joblib'))
                    if class_model_files:
                        latest_model = max(class_model_files, key=lambda x: x.stat().st_mtime)
                        
                        try:
                            model_data = joblib.load(latest_model)
                            model = model_data['model']
                            
                            X_curr = curr_features[feature_names]
                            y_curr = curr_features['Market_Sentiment']
                            
                            predictions = model.predict(X_curr)
                            accuracy = accuracy_score(y_curr, predictions)
                            
                            evaluation_results['model_performance'][f'{version}_classification'] = {
                                'accuracy': float(accuracy),
                                'samples': len(y_curr),
                                'model_file': latest_model.name,
                                'prediction_distribution': pd.Series(predictions).value_counts().to_dict()
                            }
                            
                            print(f"   ✅ {version} classification: {accuracy:.4f} accuracy")
                            
                        except Exception as e:
                            print(f"   ❌ {version} classification error: {e}")
                    
                    # Evaluate regression models
                    reg_model_files = list(self.models_dir.glob(f'*{version}_regression*.joblib'))
                    if reg_model_files and 'VIX_Target' in curr_features.columns:
                        latest_model = max(reg_model_files, key=lambda x: x.stat().st_mtime)
                        
                        try:
                            model_data = joblib.load(latest_model)
                            model = model_data['model']
                            
                            X_curr = curr_features[feature_names]
                            y_curr = curr_features['VIX_Target']
                            
                            predictions = model.predict(X_curr)
                            r2 = r2_score(y_curr, predictions)
                            rmse = np.sqrt(mean_squared_error(y_curr, predictions))
                            
                            evaluation_results['model_performance'][f'{version}_regression'] = {
                                'r2_score': float(r2),
                                'rmse': float(rmse),
                                'samples': len(y_curr),
                                'model_file': latest_model.name
                            }
                            
                            print(f"   ✅ {version} regression: R²={r2:.4f}, RMSE={rmse:.4f}")
                            
                        except Exception as e:
                            print(f"   ❌ {version} regression error: {e}")
                
                except Exception as e:
                    print(f"   ❌ {version} processing error: {e}")
        
        except ImportError as e:
            print(f"⚠️ Feature store not available: {e}")
        
        return evaluation_results
    
    def create_monitoring_visualizations(self, 
                                       drift_results: Dict,
                                       evaluation_results: Dict,
                                       reference_data: pd.DataFrame,
                                       current_data: pd.DataFrame) -> str:
        """Create monitoring dashboard visualizations"""
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Fear & Greed Classifier - Monitoring Dashboard', fontsize=16, fontweight='bold')
        
        # Plot 1: Data drift overview
        if drift_results.get('drift_details'):
            features = list(drift_results['drift_details'].keys())[:10]  # Top 10
            p_values = [drift_results['drift_details'][f]['ks_pvalue'] for f in features]
            colors = ['red' if p < 0.05 else 'green' for p in p_values]
            
            axes[0,0].barh(range(len(features)), [-np.log10(p) for p in p_values], color=colors, alpha=0.7)
            axes[0,0].axvline(x=-np.log10(0.05), color='black', linestyle='--', label='p=0.05')
            axes[0,0].set_yticks(range(len(features)))
            axes[0,0].set_yticklabels([f[:15] for f in features])
            axes[0,0].set_xlabel('-log10(p-value)')
            axes[0,0].set_title('Feature Drift Detection')
            axes[0,0].legend()
        
        # Plot 2: Target distribution comparison
        if 'Market_Sentiment' in reference_data.columns:
            ref_dist = reference_data['Market_Sentiment'].value_counts()
            curr_dist = current_data['Market_Sentiment'].value_counts()
            
            categories = list(set(ref_dist.index) | set(curr_dist.index))
            ref_counts = [ref_dist.get(cat, 0) for cat in categories]
            curr_counts = [curr_dist.get(cat, 0) for cat in categories]
            
            x = np.arange(len(categories))
            width = 0.35
            
            axes[0,1].bar(x - width/2, ref_counts, width, label='Reference', alpha=0.7)
            axes[0,1].bar(x + width/2, curr_counts, width, label='Current', alpha=0.7)
            axes[0,1].set_xlabel('Market Sentiment')
            axes[0,1].set_ylabel('Count')
            axes[0,1].set_title('Target Distribution')
            axes[0,1].set_xticks(x)
            axes[0,1].set_xticklabels(categories)
            axes[0,1].legend()
        
        # Plot 3: Model performance comparison
        model_perf = evaluation_results.get('model_performance', {})
        if model_perf:
            class_models = [k for k in model_perf.keys() if 'classification' in k]
            reg_models = [k for k in model_perf.keys() if 'regression' in k]
            
            if class_models:
                accuracies = [model_perf[m]['accuracy'] for m in class_models]
                axes[0,2].bar(range(len(class_models)), accuracies, alpha=0.7, color='skyblue')
                axes[0,2].set_ylabel('Accuracy')
                axes[0,2].set_title('Classification Performance')
                axes[0,2].set_xticks(range(len(class_models)))
                axes[0,2].set_xticklabels([m.split('_')[0] for m in class_models])
                axes[0,2].set_ylim(0, 1)
        
        # Plot 4: Feature mean shifts
        if drift_results.get('drift_details'):
            features = list(drift_results['drift_details'].keys())[:8]
            mean_shifts = [drift_results['drift_details'][f]['mean_shift_pct'] for f in features]
            colors = ['red' if abs(shift) > 5 else 'blue' for shift in mean_shifts]
            
            axes[1,0].bar(range(len(features)), mean_shifts, color=colors, alpha=0.7)
            axes[1,0].set_ylabel('Mean Shift (%)')
            axes[1,0].set_title('Feature Mean Shifts')
            axes[1,0].set_xticks(range(len(features)))
            axes[1,0].set_xticklabels([f[:10] for f in features], rotation=45)
            axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 5: VIX comparison
        if 'VIX_Price' in reference_data.columns and 'VIX_Price' in current_data.columns:
            axes[1,1].hist(reference_data['VIX_Price'].dropna(), bins=20, alpha=0.7, label='Reference', density=True)
            axes[1,1].hist(current_data['VIX_Price'].dropna(), bins=20, alpha=0.7, label='Current', density=True)
            axes[1,1].set_xlabel('VIX Price')
            axes[1,1].set_ylabel('Density')
            axes[1,1].set_title('VIX Distribution Comparison')
            axes[1,1].legend()
        
        # Plot 6: Summary metrics
        summary_data = {
            'Features with Drift': drift_results.get('features_with_drift', 0),
            'Total Features': drift_results.get('features_tested', 0),
            'Models Evaluated': len(evaluation_results.get('model_performance', {})),
            'Data Quality Score': 100 - (drift_results.get('features_with_drift', 0) / max(drift_results.get('features_tested', 1), 1) * 100)
        }
        
        labels = list(summary_data.keys())
        values = list(summary_data.values())
        
        axes[1,2].bar(range(len(labels)), values, alpha=0.7, color='lightgreen')
        axes[1,2].set_ylabel('Count/Score')
        axes[1,2].set_title('Monitoring Summary')
        axes[1,2].set_xticks(range(len(labels)))
        axes[1,2].set_xticklabels(labels, rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.monitoring_dir / f"monitoring_dashboard_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Monitoring dashboard saved: {plot_file}")
        return str(plot_file)
    
    def save_reports(self, 
                    drift_results: Dict,
                    evaluation_results: Dict,
                    evidently_reports: Dict) -> Dict[str, str]:
        """Save all monitoring reports"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        # Save drift analysis
        drift_file = self.monitoring_dir / f"drift_analysis_{timestamp}.json"
        with open(drift_file, 'w') as f:
            json.dump(drift_results, f, indent=2, default=str)
        saved_files['drift_analysis'] = str(drift_file)
        print(f"💾 Drift analysis saved: {drift_file}")
        
        # Save model evaluation
        eval_file = self.monitoring_dir / f"model_evaluation_{timestamp}.json"
        with open(eval_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        saved_files['model_evaluation'] = str(eval_file)
        print(f"💾 Model evaluation saved: {eval_file}")
        
        # Save Evidently reports
        for report_name, report in evidently_reports.items():
            try:
                report_file = self.monitoring_dir / f"evidently_{report_name}_{timestamp}.html"
                report.save(str(report_file))
                saved_files[f'evidently_{report_name}'] = str(report_file)
                print(f"💾 Evidently {report_name} saved: {report_file}")
            except Exception as e:
                print(f"⚠️ Error saving {report_name}: {e}")
        
        return saved_files
    
    def run_complete_monitoring(self) -> Dict:
        """Run complete monitoring pipeline"""
        
        print("🚀 Starting Complete Evidently Monitoring Pipeline V2")
        print("=" * 60)
        
        # Load data
        reference_data, current_data = self.load_holdout_data()
        
        # Manual drift detection
        print("\n📊 Detecting data drift...")
        drift_results = self.manual_drift_detection(reference_data, current_data)
        
        # Create Evidently reports
        print("\n📋 Creating Evidently reports...")
        evidently_reports = self.create_evidently_reports(reference_data, current_data)
        
        # Evaluate models
        print("\n🤖 Evaluating model performance...")
        evaluation_results = self.evaluate_models(reference_data, current_data)
        
        # Create visualizations
        print("\n📊 Creating monitoring dashboard...")
        dashboard_file = self.create_monitoring_visualizations(
            drift_results, evaluation_results, reference_data, current_data
        )
        
        # Save reports
        print("\n💾 Saving monitoring reports...")
        saved_files = self.save_reports(drift_results, evaluation_results, evidently_reports)
        saved_files['dashboard'] = dashboard_file
        
        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'drift_summary': drift_results.get('summary', {}),
            'model_performance': evaluation_results.get('model_performance', {}),
            'saved_files': saved_files,
            'alerts': []
        }
        
        # Generate alerts
        if drift_results.get('summary', {}).get('overall_drift_detected'):
            summary['alerts'].append("🚨 Data drift detected!")
        
        if drift_results.get('target_drift', {}).get('drift_detected'):
            summary['alerts'].append("🚨 Target distribution drift detected!")
        
        drift_pct = drift_results.get('summary', {}).get('drift_percentage', 0)
        if drift_pct > 20:
            summary['alerts'].append(f"⚠️ High drift percentage: {drift_pct:.1f}%")
        
        # Save summary
        summary_file = self.monitoring_dir / f"monitoring_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n✅ Complete monitoring pipeline finished!")
        print(f"📄 Summary saved: {summary_file}")
        print(f"📁 All reports in: {self.monitoring_dir}")
        
        # Print key findings
        print(f"\n📈 Key Findings:")
        print(f"   Drift percentage: {drift_pct:.1f}%")
        print(f"   Features with drift: {drift_results.get('features_with_drift', 0)}/{drift_results.get('features_tested', 0)}")
        print(f"   Models evaluated: {len(evaluation_results.get('model_performance', {}))}")
        
        if summary.get('alerts'):
            print("⚠️ Alerts:")
            for alert in summary['alerts']:
                print(f"   {alert}")
        else:
            print("✅ No significant issues detected")
        
        return summary


def main():
    """Main function"""
    monitor = FearGreedEvidentlyMonitorV2()
    results = monitor.run_complete_monitoring()
    
    print(f"\n🎉 Monitoring completed successfully!")
    return results


if __name__ == "__main__":
    results = main()
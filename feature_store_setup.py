# feature_store_setup.py

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import joblib
import json
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class FearGreedFeatureStore:
    """
    Feature Store for Fear & Greed Classifier
    Integrates with your existing data structure and teammate's feature engineering
    """
    
    def __init__(self, data_dir: str = "data", feature_store_dir: str = "feature_store"):
        self.data_dir = Path(data_dir)
        self.feature_store_dir = Path(feature_store_dir)
        self.feature_store_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature metadata
        self.feature_metadata = {}
        self.feature_versions = {}
        
        # Load existing dataset
        self.dataset_path = Path("Market-Fear-Greed-Indicator/Data.csv")
        self.feature_summary_path = self.data_dir / "processed" / "feature_summary.csv"
        
    def load_base_dataset(self) -> pd.DataFrame:
        """Load the merged dataset created by your data pipeline"""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")
        
        df = pd.read_csv(self.dataset_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        print(f"✅ Loaded base dataset: {df.shape[0]} rows × {df.shape[1]} features")
        return df
    
    def load_feature_metadata(self) -> pd.DataFrame:
        """Load feature metadata created by your data pipeline"""
        if not self.feature_summary_path.exists():
            print("⚠️ Feature summary not found. Returning empty metadata.")
            return pd.DataFrame(columns=['Category', 'Feature'])
        
        metadata = pd.read_csv(self.feature_summary_path)
        print(f"✅ Loaded feature metadata: {len(metadata)} features documented")
        return metadata
    
    def create_feature_groups(self, df: pd.DataFrame, metadata: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Organize features into logical groups based on your existing categorization
        """
        feature_groups = {}
        
        # Group by category from metadata
        for category in metadata['Category'].unique():
            if pd.isna(category):
                continue
            
            category_features = metadata[metadata['Category'] == category]['Feature'].tolist()
            # Only include features that exist in the dataset
            category_features = [f for f in category_features if f in df.columns]
            
            if category_features:
                feature_groups[category.lower().replace(' ', '_')] = category_features
        
        # Add derived features group
        derived_features = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                           ['ratio', 'slope', 'fear_level', 'appetite', 'rotation'])]
        if derived_features:
            feature_groups['derived_indicators'] = derived_features
        
        return feature_groups
    
    def implement_v1_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Implement your teammate's v1 feature engineering: percentage returns
        """
        df_v1 = df.copy()
        v1_features = []
        
        # Get price columns (excluding VIX to avoid leakage)
        price_cols = [col for col in df.columns if col.endswith('_Price') and 'VIX' not in col]
        
        for col in price_cols:
            ret_col = f"{col}_ret_pct"
            df_v1[ret_col] = df_v1[col].pct_change() * 100.0
            v1_features.append(ret_col)
        
        # Add existing non-price numeric features
        existing_features = [col for col in df.columns if 
                           col not in price_cols and 
                           'VIX' not in col and
                           df[col].dtype in ['float64', 'int64'] and
                           col not in ['Market_Sentiment']]
        
        v1_features.extend(existing_features)
        v1_features = [f for f in v1_features if f in df_v1.columns]
        
        print(f"✅ V1 Features: {len(v1_features)} features (percentage returns)")
        return df_v1, v1_features
    
    def implement_v2_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Implement your teammate's v2 feature engineering: v1 + moving averages
        """
        df_v2, v1_features = self.implement_v1_features(df)
        
        # Add moving averages for price columns (excluding VIX)
        price_cols = [col for col in df.columns if col.endswith('_Price') and 'VIX' not in col]
        ma_features = []
        
        for col in price_cols:
            # 7-day moving average
            ma7_col = f"{col}_ma7"
            df_v2[ma7_col] = df_v2[col].rolling(window=7, min_periods=1).mean()
            ma_features.append(ma7_col)
            
            # 30-day moving average
            ma30_col = f"{col}_ma30"
            df_v2[ma30_col] = df_v2[col].rolling(window=30, min_periods=1).mean()
            ma_features.append(ma30_col)
        
        v2_features = v1_features + ma_features
        v2_features = [f for f in v2_features if f in df_v2.columns]
        
        print(f"✅ V2 Features: {len(v2_features)} features (returns + moving averages)")
        return df_v2, v2_features
    
    def implement_v3_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Implement your teammate's v3 feature engineering: v2 + ratios
        """
        df_v3, v2_features = self.implement_v2_features(df)
        
        # Key tickers for ratio creation (excluding VIX)
        key_tickers = ['SPY', 'GLD', 'QQQ', 'IWM', 'DIA', 'TLT']
        
        # Price ratios
        price_cols = [col for col in df.columns if col.endswith('_Price') and 
                     any(ticker in col for ticker in key_tickers)]
        
        ratio_features = []
        
        # Price/Price ratios
        for i in range(len(price_cols)):
            for j in range(i + 1, len(price_cols)):
                col_a, col_b = price_cols[i], price_cols[j]
                ratio_col = f"ratio_{col_a}_over_{col_b}"
                df_v3[ratio_col] = df_v3[col_a] / (df_v3[col_b] + 1e-9)
                ratio_features.append(ratio_col)
        
        # Volume ratios (if available)
        volume_cols = [col for col in df.columns if col.endswith('_Volume') and 
                      any(ticker in col for ticker in key_tickers)]
        
        for i in range(len(volume_cols)):
            for j in range(i + 1, len(volume_cols)):
                col_a, col_b = volume_cols[i], volume_cols[j]
                ratio_col = f"ratio_{col_a}_over_{col_b}"
                df_v3[ratio_col] = df_v3[col_a] / (df_v3[col_b] + 1e-9)
                ratio_features.append(ratio_col)
        
        v3_features = v2_features + ratio_features
        v3_features = [f for f in v3_features if f in df_v3.columns]
        
        print(f"✅ V3 Features: {len(v3_features)} features (returns + MA + ratios)")
        return df_v3, v3_features
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variable following your teammate's approach
        """
        df_target = df.copy()
        
        if 'VIX_Price' not in df.columns:
            raise ValueError("VIX_Price column not found for target creation")
        
        vix_prices = df['VIX_Price'].astype(float)
        mu = vix_prices.mean()
        sigma = vix_prices.std()
        
        def vix_to_sentiment(v):
            if pd.isna(v):
                return np.nan
            if v > mu + sigma:
                return "fear"    # High VIX = Fear
            if v < mu - sigma:
                return "greed"   # Low VIX = Greed
            return "stable"
        
        df_target['Market_Sentiment'] = vix_prices.apply(vix_to_sentiment)
        df_target['VIX_Target'] = vix_prices  # For regression tasks
        
        print(f"✅ Target variables created:")
        print(f"   - Market_Sentiment: {df_target['Market_Sentiment'].value_counts().to_dict()}")
        print(f"   - VIX_Target: continuous values (μ={mu:.2f}, σ={sigma:.2f})")
        
        return df_target
    
    def save_feature_set(self, df: pd.DataFrame, features: List[str], version: str, 
                        description: str) -> str:
        """
        Save a feature set to the feature store
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        feature_set_name = f"fear_greed_{version}_{timestamp}"
        
        # Save features data
        feature_data = df[features + ['VIX_Target', 'Market_Sentiment']].copy()
        feature_file = self.feature_store_dir / f"{feature_set_name}.parquet"
        feature_data.to_parquet(feature_file)
        
        # Save metadata
        metadata = {
            'name': feature_set_name,
            'version': version,
            'description': description,
            'created_at': timestamp,
            'num_features': len(features),
            'num_samples': len(df),
            'features': features,
            'target_columns': ['VIX_Target', 'Market_Sentiment'],
            'date_range': {
                'start': str(df.index.min().date()),
                'end': str(df.index.max().date())
            }
        }
        
        metadata_file = self.feature_store_dir / f"{feature_set_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved feature set: {feature_set_name}")
        print(f"   - Data: {feature_file}")
        print(f"   - Metadata: {metadata_file}")
        
        return feature_set_name
    
    def list_feature_sets(self) -> pd.DataFrame:
        """
        List all available feature sets
        """
        metadata_files = list(self.feature_store_dir.glob("*_metadata.json"))
        
        if not metadata_files:
            print("No feature sets found")
            return pd.DataFrame()
        
        feature_sets = []
        for metadata_file in metadata_files:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            feature_sets.append({
                'Name': metadata['name'],
                'Version': metadata['version'],
                'Features': metadata['num_features'],
                'Samples': metadata['num_samples'],
                'Created': metadata['created_at'],
                'Description': metadata['description']
            })
        
        return pd.DataFrame(feature_sets).sort_values('Created', ascending=False)
    
    def load_feature_set(self, feature_set_name: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Load a feature set from the feature store
        """
        feature_file = self.feature_store_dir / f"{feature_set_name}.parquet"
        metadata_file = self.feature_store_dir / f"{feature_set_name}_metadata.json"
        
        if not feature_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"Feature set {feature_set_name} not found")
        
        df = pd.read_parquet(feature_file)
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"✅ Loaded feature set: {feature_set_name}")
        print(f"   - Shape: {df.shape}")
        print(f"   - Features: {len(metadata['features'])}")
        
        return df, metadata

def main():
    """
    Main function to set up the feature store with your existing data
    """
    print("🚀 Setting up Fear & Greed Feature Store")
    print("=" * 60)
    
    # Initialize feature store
    fs = FearGreedFeatureStore()
    
    # Load your existing data
    df = fs.load_base_dataset()
    metadata = fs.load_feature_metadata()
    
    # Create feature groups
    feature_groups = fs.create_feature_groups(df, metadata)
    print(f"\n📊 Feature Groups Created:")
    for group, features in feature_groups.items():
        print(f"   - {group}: {len(features)} features")
    
    # Add target variables
    df_with_targets = fs.create_target_variable(df)
    
    # Create and save all three feature versions
    print(f"\n🔧 Creating Feature Versions...")
    
    # V1: Returns
    df_v1, v1_features = fs.implement_v1_features(df_with_targets)
    v1_name = fs.save_feature_set(
        df_v1, v1_features, "v1", 
        "Percentage daily returns for all non-VIX price columns"
    )
    
    # V2: Returns + Moving Averages
    df_v2, v2_features = fs.implement_v2_features(df_with_targets)
    v2_name = fs.save_feature_set(
        df_v2, v2_features, "v2",
        "V1 features + 7-day and 30-day moving averages"
    )
    
    # V3: Returns + MA + Ratios
    df_v3, v3_features = fs.implement_v3_features(df_with_targets)
    v3_name = fs.save_feature_set(
        df_v3, v3_features, "v3",
        "V2 features + price/price and volume/volume ratios"
    )
    
    # List all feature sets
    print(f"\n📋 Available Feature Sets:")
    feature_sets_df = fs.list_feature_sets()
    print(feature_sets_df.to_string(index=False))
    
    print(f"\n🎉 Feature Store Setup Complete!")
    print(f"   - Feature store location: {fs.feature_store_dir.resolve()}")
    print(f"   - Ready for AutoML integration")
    
    return fs, feature_sets_df

if __name__ == "__main__":
    fs, feature_sets = main()

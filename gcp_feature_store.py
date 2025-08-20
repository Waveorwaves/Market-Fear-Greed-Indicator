# gcp_feature_store.py

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# GCP imports
try:
    from google.cloud import storage
    from google.cloud import bigquery
    from google.cloud import aiplatform
    from google.oauth2 import service_account
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    print("⚠️  GCP libraries not installed. Install with:")
    print("   pip install google-cloud-storage google-cloud-bigquery google-cloud-aiplatform")

# Local imports
from feature_store_setup import FearGreedFeatureStore

class GCPFearGreedFeatureStore(FearGreedFeatureStore):
    """
    GCP-integrated Feature Store for Fear & Greed Classifier
    Extends the local feature store with Google Cloud Platform capabilities
    """
    
    def __init__(self, 
                 project_id: str,
                 bucket_name: str,
                 dataset_id: str = "fear_greed_ml",
                 region: str = "us-central1",
                 credentials_path: Optional[str] = None,
                 local_data_dir: str = "data",
                 local_feature_store_dir: str = "feature_store"):
        
        # Initialize parent class
        super().__init__(local_data_dir, local_feature_store_dir)
        
        # GCP configuration
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.dataset_id = dataset_id
        self.region = region
        
        # Initialize GCP clients
        if GCP_AVAILABLE:
            self._init_gcp_clients(credentials_path)
        else:
            raise ImportError("GCP libraries not available. Please install required packages.")
        
        # GCP paths
        self.gcs_feature_store_path = f"gs://{bucket_name}/feature_store/"
        self.bq_table_prefix = f"{project_id}.{dataset_id}"
        
        print(f"🌐 GCP Feature Store initialized:")
        print(f"   Project: {project_id}")
        print(f"   Bucket: {bucket_name}")
        print(f"   Dataset: {dataset_id}")
        print(f"   Region: {region}")
    
    def _init_gcp_clients(self, credentials_path: Optional[str] = None):
        """Initialize GCP clients with optional service account credentials"""
        
        if credentials_path and os.path.exists(credentials_path):
            # Use service account credentials
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
            self.storage_client = storage.Client(credentials=credentials, project=self.project_id)
            self.bq_client = bigquery.Client(credentials=credentials, project=self.project_id)
            print(f"✅ Using service account credentials: {credentials_path}")
        else:
            # Use default credentials (gcloud auth application-default login)
            self.storage_client = storage.Client(project=self.project_id)
            self.bq_client = bigquery.Client(project=self.project_id)
            print(f"✅ Using default GCP credentials")
        
        # Initialize Vertex AI
        aiplatform.init(project=self.project_id, location=self.region)
    
    def setup_gcp_infrastructure(self):
        """Set up required GCP infrastructure"""
        
        print("🏗️  Setting up GCP infrastructure...")
        print("-" * 50)
        
        # Create GCS bucket if it doesn't exist
        self._create_gcs_bucket()
        
        # Create BigQuery dataset if it doesn't exist
        self._create_bq_dataset()
        
        # Create feature store directories in GCS
        self._create_gcs_directories()
        
        print("✅ GCP infrastructure setup complete!")
    
    def _create_gcs_bucket(self):
        """Create GCS bucket for feature store"""
        
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            if not bucket.exists():
                bucket = self.storage_client.create_bucket(
                    self.bucket_name, 
                    location=self.region
                )
                print(f"✅ Created GCS bucket: {self.bucket_name}")
            else:
                print(f"✅ GCS bucket exists: {self.bucket_name}")
        except Exception as e:
            print(f"❌ Failed to create/check GCS bucket: {e}")
    
    def _create_bq_dataset(self):
        """Create BigQuery dataset for feature metadata"""
        
        try:
            dataset_id = f"{self.project_id}.{self.dataset_id}"
            dataset = bigquery.Dataset(dataset_id)
            dataset.location = self.region
            
            try:
                dataset = self.bq_client.create_dataset(dataset, timeout=30)
                print(f"✅ Created BigQuery dataset: {dataset_id}")
            except Exception:
                # Dataset might already exist
                print(f"✅ BigQuery dataset exists: {dataset_id}")
                
        except Exception as e:
            print(f"❌ Failed to create/check BigQuery dataset: {e}")
    
    def _create_gcs_directories(self):
        """Create directory structure in GCS"""
        
        directories = [
            "feature_store/datasets/",
            "feature_store/metadata/", 
            "feature_store/schemas/",
            "models/",
            "experiments/"
        ]
        
        for directory in directories:
            try:
                # Create a placeholder file to establish the directory
                blob_name = f"{directory}.gitkeep"
                bucket = self.storage_client.bucket(self.bucket_name)
                blob = bucket.blob(blob_name)
                blob.upload_from_string("")
                print(f"✅ Created GCS directory: gs://{self.bucket_name}/{directory}")
            except Exception as e:
                print(f"⚠️  Directory creation warning for {directory}: {e}")
    
    def upload_to_gcs(self, local_file_path: str, gcs_blob_name: str) -> str:
        """Upload file to Google Cloud Storage"""
        
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(gcs_blob_name)
            blob.upload_from_filename(local_file_path)
            
            gcs_path = f"gs://{self.bucket_name}/{gcs_blob_name}"
            print(f"✅ Uploaded to GCS: {gcs_path}")
            return gcs_path
            
        except Exception as e:
            print(f"❌ Failed to upload {local_file_path}: {e}")
            return ""
    
    def download_from_gcs(self, gcs_blob_name: str, local_file_path: str) -> bool:
        """Download file from Google Cloud Storage"""
        
        try:
            bucket = self.storage_client.bucket(self.bucket_name)
            blob = bucket.blob(gcs_blob_name)
            blob.download_to_filename(local_file_path)
            
            print(f"✅ Downloaded from GCS: {local_file_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {gcs_blob_name}: {e}")
            return False
    
    def save_feature_set_to_gcp(self, df: pd.DataFrame, features: List[str], 
                               version: str, description: str) -> Dict[str, str]:
        """Save feature set to both local storage and GCP"""
        
        print(f"💾 Saving feature set to GCP...")
        
        # First save locally using parent method
        feature_set_name = self.save_feature_set(df, features, version, description)
        
        # Upload to GCS
        local_parquet = self.feature_store_dir / f"{feature_set_name}.parquet"
        local_metadata = self.feature_store_dir / f"{feature_set_name}_metadata.json"
        
        gcs_paths = {}
        
        # Upload parquet file
        if local_parquet.exists():
            gcs_blob_name = f"feature_store/datasets/{feature_set_name}.parquet"
            gcs_paths['data'] = self.upload_to_gcs(str(local_parquet), gcs_blob_name)
        
        # Upload metadata
        if local_metadata.exists():
            gcs_blob_name = f"feature_store/metadata/{feature_set_name}_metadata.json"
            gcs_paths['metadata'] = self.upload_to_gcs(str(local_metadata), gcs_blob_name)
        
        # Save to BigQuery for querying
        self._save_feature_metadata_to_bq(feature_set_name, local_metadata)
        
        # Create feature set registry entry
        self._register_feature_set(feature_set_name, gcs_paths, version, description)
        
        return {
            'feature_set_name': feature_set_name,
            'gcs_paths': gcs_paths,
            'local_paths': {
                'data': str(local_parquet),
                'metadata': str(local_metadata)
            }
        }
    
    def _save_feature_metadata_to_bq(self, feature_set_name: str, metadata_file: Path):
        """Save feature metadata to BigQuery for easy querying"""
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Prepare data for BigQuery
            bq_data = {
                'feature_set_name': feature_set_name,
                'version': metadata['version'],
                'description': metadata['description'],
                'created_at': datetime.now().isoformat(),
                'num_features': metadata['num_features'],
                'num_samples': metadata['num_samples'],
                'date_range_start': metadata['date_range']['start'],
                'date_range_end': metadata['date_range']['end'],
                'features_json': json.dumps(metadata['features'])
            }
            
            # Create table if it doesn't exist
            table_id = f"{self.bq_table_prefix}.feature_sets_registry"
            
            # Define schema
            schema = [
                bigquery.SchemaField("feature_set_name", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("version", "STRING", mode="REQUIRED"),
                bigquery.SchemaField("description", "STRING"),
                bigquery.SchemaField("created_at", "TIMESTAMP"),
                bigquery.SchemaField("num_features", "INTEGER"),
                bigquery.SchemaField("num_samples", "INTEGER"),
                bigquery.SchemaField("date_range_start", "STRING"),
                bigquery.SchemaField("date_range_end", "STRING"),
                bigquery.SchemaField("features_json", "STRING"),
            ]
            
            # Create table if it doesn't exist
            try:
                table = bigquery.Table(table_id, schema=schema)
                table = self.bq_client.create_table(table)
                print(f"✅ Created BigQuery table: {table_id}")
            except Exception:
                # Table already exists
                pass
            
            # Insert data
            table = self.bq_client.get_table(table_id)
            rows_to_insert = [bq_data]
            errors = self.bq_client.insert_rows_json(table, rows_to_insert)
            
            if errors:
                print(f"❌ BigQuery insert errors: {errors}")
            else:
                print(f"✅ Saved metadata to BigQuery: {table_id}")
                
        except Exception as e:
            print(f"❌ Failed to save metadata to BigQuery: {e}")
    
    def _register_feature_set(self, feature_set_name: str, gcs_paths: Dict[str, str], 
                             version: str, description: str):
        """Register feature set in a central registry"""
        
        registry_file = self.feature_store_dir / "feature_registry.json"
        
        # Load existing registry or create new
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                registry = json.load(f)
        else:
            registry = {"feature_sets": []}
        
        # Add new entry
        entry = {
            "name": feature_set_name,
            "version": version,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "gcs_paths": gcs_paths,
            "status": "active"
        }
        
        registry["feature_sets"].append(entry)
        
        # Save registry locally
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
        
        # Upload registry to GCS
        gcs_registry_path = "feature_store/feature_registry.json"
        self.upload_to_gcs(str(registry_file), gcs_registry_path)
        
        print(f"✅ Registered feature set: {feature_set_name}")
    
    def load_feature_set_from_gcp(self, feature_set_name: str, 
                                 prefer_gcs: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """Load feature set from GCP or local storage"""
        
        if prefer_gcs:
            # Try to download from GCS first
            gcs_blob_name = f"feature_store/datasets/{feature_set_name}.parquet"
            local_file = self.feature_store_dir / f"{feature_set_name}.parquet"
            
            if self.download_from_gcs(gcs_blob_name, str(local_file)):
                # Also download metadata
                gcs_metadata_blob = f"feature_store/metadata/{feature_set_name}_metadata.json"
                local_metadata = self.feature_store_dir / f"{feature_set_name}_metadata.json"
                self.download_from_gcs(gcs_metadata_blob, str(local_metadata))
        
        # Load using parent method
        return self.load_feature_set(feature_set_name)
    
    def list_feature_sets_from_bq(self) -> pd.DataFrame:
        """List feature sets from BigQuery registry"""
        
        try:
            query = f"""
            SELECT 
                feature_set_name,
                version,
                description,
                created_at,
                num_features,
                num_samples,
                date_range_start,
                date_range_end
            FROM `{self.bq_table_prefix}.feature_sets_registry`
            ORDER BY created_at DESC
            """
            
            df = self.bq_client.query(query).to_dataframe()
            print(f"✅ Retrieved {len(df)} feature sets from BigQuery")
            return df
            
        except Exception as e:
            print(f"❌ Failed to query BigQuery registry: {e}")
            # Fallback to local method
            return self.list_feature_sets()
    
    def sync_with_git(self, commit_message: str = None):
        """Sync feature store with Git repository"""
        
        if not commit_message:
            commit_message = f"Update feature store - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        print("🔄 Syncing with Git...")
        
        try:
            import subprocess
            
            # Add feature store files to git
            subprocess.run(["git", "add", str(self.feature_store_dir)], check=True)
            
            # Add any new model files
            if Path("models").exists():
                subprocess.run(["git", "add", "models/"], check=True)
            
            # Commit changes
            result = subprocess.run(
                ["git", "commit", "-m", commit_message],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                print(f"✅ Git commit successful: {commit_message}")
                
                # Optionally push to remote
                push_result = subprocess.run(
                    ["git", "push"],
                    capture_output=True, text=True
                )
                
                if push_result.returncode == 0:
                    print("✅ Git push successful")
                else:
                    print(f"⚠️  Git push failed: {push_result.stderr}")
                    
            else:
                if "nothing to commit" in result.stdout:
                    print("ℹ️  No changes to commit")
                else:
                    print(f"❌ Git commit failed: {result.stderr}")
                    
        except subprocess.CalledProcessError as e:
            print(f"❌ Git operation failed: {e}")
        except FileNotFoundError:
            print("❌ Git not found. Please ensure Git is installed and in PATH")
    
def create_vertex_ai_feature_store(self, feature_store_id: str = "fear-greed-fs"):
    """Create a Vertex AI Feature Store (optional advanced feature)"""
    
    print(f"🧠 Creating Vertex AI Feature Store: {feature_store_id}")
    
    try:
             # This is for advanced users who want to use Vertex AI Feature Store
             # For now, we'll just print the command they would need to run
        print("📋 To create Vertex AI Feature Store, run:")
        print(f"gcloud ai feature-stores create {feature_store_id} \\\\")
        print(f"    --region={self.region} \\\\")
        print(f"    --project={self.project_id}")
    
        print("\\n💡 This is optional - your current setup works great without it!")
    
    except Exception as e:
        print(f"❌ Vertex AI Feature Store creation info:  {e}")

def main():
    """
    Main function to set up GCP-integrated feature store
    """
    print("🌐 GCP Fear & Greed Feature Store Setup")
    print("=" * 60)
    
    # Configuration - UPDATED FOR YOUR PROJECT
    config = {
        'project_id': 'mlo-project-finance-indicator',
        'bucket_name': 'mlo-project-finance-indicator-ml-data',
        'dataset_id': 'fear_greed_ml',
        'region': 'us-central1',
        'credentials_path': 'sa-credentials.json',
        'local_data_dir': 'Market-Fear-Greed-Indicator/data'
    }
    
    print("✅ Configuration updated for your project:")
    print(f"   project_id: {config['project_id']}")
    print(f"   bucket_name: {config['bucket_name']}")
    print()
    
    try:
        # Initialize GCP Feature Store
        gcp_fs = GCPFearGreedFeatureStore(**config)
        
        # Setup GCP infrastructure
        gcp_fs.setup_gcp_infrastructure()
        
        # Load and process data (same as local version)
        df = gcp_fs.load_base_dataset()
        metadata = gcp_fs.load_feature_metadata()
        
        # Create feature groups
        feature_groups = gcp_fs.create_feature_groups(df, metadata)
        print(f"\n📊 Feature Groups Created:")
        for group, features in feature_groups.items():
            print(f"   - {group}: {len(features)} features")
        
        # Add target variables
        df_with_targets = gcp_fs.create_target_variable(df)
        
        # Create and save all three feature versions to GCP
        print(f"\n🔧 Creating Feature Versions and Uploading to GCP...")
        
        # V1: Returns
        df_v1, v1_features = gcp_fs.implement_v1_features(df_with_targets)
        v1_result = gcp_fs.save_feature_set_to_gcp(
            df_v1, v1_features, "v1", 
            "Percentage daily returns for all non-VIX price columns"
        )
        
        # V2: Returns + Moving Averages
        df_v2, v2_features = gcp_fs.implement_v2_features(df_with_targets)
        v2_result = gcp_fs.save_feature_set_to_gcp(
            df_v2, v2_features, "v2",
            "V1 features + 7-day and 30-day moving averages"
        )
        
        # V3: Returns + MA + Ratios
        df_v3, v3_features = gcp_fs.implement_v3_features(df_with_targets)
        v3_result = gcp_fs.save_feature_set_to_gcp(
            df_v3, v3_features, "v3",
            "V2 features + price/price and volume/volume ratios"
        )
        
        # List feature sets from BigQuery
        print(f"\n📋 Feature Sets in BigQuery:")
        try:
            feature_sets_df = gcp_fs.list_feature_sets_from_bq()
            print(feature_sets_df.to_string(index=False))
        except Exception as e:
            print(f"⚠️  BigQuery listing failed, using local: {e}")
            feature_sets_df = gcp_fs.list_feature_sets()
            print(feature_sets_df.to_string(index=False))
        
        # Sync with Git
        gcp_fs.sync_with_git("Initial GCP feature store setup")
        
        print(f"\n🎉 GCP Feature Store Setup Complete!")
        print(f"   - Local feature store: {gcp_fs.feature_store_dir.resolve()}")
        print(f"   - GCS bucket: gs://{config['bucket_name']}/feature_store/")
        print(f"   - BigQuery dataset: {config['project_id']}.{config['dataset_id']}")
        print(f"   - Git repository: synced")
        
        print(f"\n🎯 Next Steps:")
        print("   1. Run AutoML experiments with GCP integration")
        print("   2. Set up model training pipelines")
        print("   3. Configure automated retraining")
        print("   4. Deploy models to Vertex AI")
        
        return gcp_fs
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check GCP credentials: gcloud auth application-default login")
        print("   2. Verify project permissions for Storage, BigQuery, and Vertex AI")
        print("   3. Ensure billing is enabled for the GCP project")
        print("   4. Check that the specified region is available")
        return None

if __name__ == "__main__":
    gcp_fs = main()

    # Example of loading a feature set from GCP
    if gcp_fs:
        print("\n🔄 Example: Loading feature set 'v3' from GCP...")
        try:
            df_loaded, metadata_loaded = gcp_fs.load_feature_set_from_gcp("v3")
            print(f"✅ Loaded 'v3' successfully: {df_loaded.shape}")
        except Exception as e:
            print(f"❌ Failed to load 'v3' from GCP: {e}")
            print("   Trying to load from local cache...")
            try:
                df_loaded, metadata_loaded = gcp_fs.load_feature_set("v3")
                print(f"✅ Loaded 'v3' from local cache: {df_loaded.shape}")
            except Exception as e_local:
                print(f"❌ Failed to load 'v3' locally: {e_local}")
                print("   Please run the setup script again.")

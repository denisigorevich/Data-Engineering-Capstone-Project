# Cloud Data Platform Capstone Project

A comprehensive data platform implementation showcasing modern cloud-native data engineering practices using Google Cloud Platform.

## 🏗️ Platform Architecture

This project demonstrates a complete data pipeline from ingestion to analytics, implementing enterprise-grade data quality, governance, and monitoring practices.

```
Raw Data → Data Ingestion → Data Quality → Data Enrichment → Data Modeling → Analytics
    ↓            ↓              ↓              ↓              ↓           ↓
 Various       GCS/BQ        Dataplex       DBT/Spark      BigQuery    Dashboards
 Sources      Pipeline        DQ Rules      Transform      Data Mart    Reports
```

## 📁 Project Structure

### Core Components

- **`data_ingestion/`** - Data ingestion pipelines and processing scripts
- **`src/data_quality/`** - Dataplex-based data quality management
- **`src/data_cleaning/`** - Data cleaning and preprocessing utilities  
- **`src/db_testing/`** - Database connectivity and validation tests
- **`config/`** - Configuration files and settings
- **`docs/`** - Documentation and data profiling reports

### Development Infrastructure

- **`.pre-commit-config.yaml`** - Pre-commit hooks for code quality
- **`.cz.toml`** - Conventional commits configuration
- **`CODEOWNERS`** - Code ownership and review requirements
- **`requirements.txt`** - Python dependencies

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Google Cloud SDK
- Access to Google Cloud Platform
- Git

### Setup

1. **Clone and setup environment:**
   ```bash
   git clone <repository-url>
   cd capstone_project
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Install development tools:**
   ```bash
   pip install pre-commit commitizen ruff bandit
   pre-commit install
   ```

3. **Configure Google Cloud:**
   ```bash
   gcloud auth application-default login
   gcloud config set project your-project-id
   ```

## 🔧 Development Workflow

This project follows enterprise development practices:

### Branch Strategy (Trunk-based Development)
- **Main branch protection**: No direct pushes
- **Feature branches**: `feat/feature-name`
- **Bug fixes**: `fix/issue-description`  
- **Documentation**: `docs/topic-name`

### Commit Standards (Conventional Commits)
```bash
feat(data-quality): add VIN validation rules
fix(ingestion): resolve connection timeout issues
docs(readme): update setup instructions
```

### Code Quality Gates
- **Pre-commit hooks**: Automated linting and formatting
- **Security scanning**: Bandit for Python security analysis
- **Code formatting**: Ruff for Python formatting
- **Branch naming**: Enforced via pre-commit hooks

## 📊 Data Pipeline Components

### 1. Data Ingestion (`data_ingestion/`)
- **Raw data processing**: Vehicle data cleaning and validation
- **Format standardization**: CSV, JSON, Parquet processing
- **Error handling**: Data validation and quality checks

### 2. Data Quality (`src/data_quality/`)
Three approaches to data quality management:

- **`dq_basic_dataplex_entity.py`**: Entity-based scanning using Dataplex lakes/zones
- **`dq_simple_bigquery_ondemand.py`**: Direct BigQuery table scanning for quick validation  
- **`dq_scheduled_with_export.py`**: Production-ready scheduled scans with result export

**Key Features:**
- VIN validation (17 characters, uniqueness)
- Completeness checks (non-null cylinders)
- Automated scheduling and alerting
- BigQuery result export for analysis

### 3. Data Processing (`src/data_cleaning/`)
- **Data transformation**: Field standardization and normalization
- **Quality remediation**: Automatic data correction where possible
- **Enrichment preparation**: Data structure optimization

### 4. Database Operations (`src/db_testing/`)
- **Connection validation**: Dataplex and BigQuery connectivity tests
- **Performance monitoring**: Query performance and optimization
- **Health checks**: System status and availability validation

## 🎯 Use Cases Demonstrated

### Data Quality Management
- **Real-time validation**: On-demand data quality scans
- **Scheduled monitoring**: Automated daily quality checks
- **Compliance reporting**: Export results for governance dashboards

### Data Engineering Best Practices
- **Infrastructure as Code**: Configuration-driven setup
- **Automated testing**: Pre-commit hooks and validation
- **Documentation**: Comprehensive READMEs and inline comments
- **Version control**: Conventional commits and branching strategy

### Cloud-Native Architecture
- **Serverless processing**: Cloud Functions for triggered workflows
- **Managed services**: BigQuery, Dataplex, Cloud Storage
- **Scalable design**: Auto-scaling and resource optimization

## 📈 Monitoring & Observability

### Data Quality Metrics
- **Completeness scores**: Percentage of non-null values
- **Validity rates**: Format and business rule compliance  
- **Uniqueness checks**: Duplicate detection and reporting
- **Consistency monitoring**: Cross-table validation

### Pipeline Monitoring
- **Execution logs**: Detailed processing logs and error tracking
- **Performance metrics**: Processing time and resource usage
- **Alerting**: Automated notifications for failures and issues

## 🔒 Security & Governance

### Access Control
- **Code ownership**: CODEOWNERS file for review requirements
- **Branch protection**: Required reviews and status checks
- **Service accounts**: Principle of least privilege

### Data Governance  
- **Quality rules**: Automated validation and enforcement
- **Audit trails**: Complete processing lineage
- **Compliance**: Data retention and privacy controls

## 🚀 Getting Started with Components

### Quick Data Quality Check
```bash
# Test connectivity
python src/db_testing/test_dataplex_connection.py

# Run simple data quality scan
python src/data_quality/dq_simple_bigquery_ondemand.py

# Schedule automated quality monitoring
python src/data_quality/dq_scheduled_with_export.py
```

### Development Workflow
```bash
# Create feature branch
git checkout -b feat/new-feature

# Make changes with proper commits
cz commit  # Interactive conventional commit

# Pre-commit hooks run automatically
git push origin feat/new-feature

# Create PR for review
```

## 📚 Documentation

- **[Data Quality Guide](src/data_quality/README_DataQuality.md)** - Comprehensive data quality setup
- **[Data Profiling Report](docs/data_profiling_report.md)** - Dataset analysis and insights

## 🤝 Contributing

1. Follow the branching strategy (trunk-based development)
2. Use conventional commits for all changes
3. Ensure pre-commit hooks pass
4. Update documentation for new features
5. Add tests for new functionality

## 📋 Roadmap

- [ ] **Data Enrichment**: Implement DBT transformations
- [ ] **Advanced Monitoring**: Add Datadog/Grafana dashboards  
- [ ] **ML Pipeline**: Integrate ML model training and serving
- [ ] **Real-time Processing**: Add streaming data capabilities
- [ ] **Multi-cloud**: Extend to AWS/Azure integrations

## 📄 License

[Add your license information here]

## 👥 Team

Data Engineering Team - @datatonic/data-eng-team

---

*This capstone project demonstrates enterprise-grade data platform implementation using modern cloud technologies and best practices.*

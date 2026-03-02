# Single Database Evidence Retrieval Workflows for Living Guideline Development and Maintenance

This repository contains code necessary to replicate experiments and analyses necessary for the paper: Automated Evidence Surveillance with AI-Enabled Pre-Ranking in Living Evidence-Based Guideline Maintenance: A Simulation Study

## Overview

This project implements and evaluates single database evidence retrieval workflows for living guideline development and maintenance. The following workflow are simulated: 

* Single database topic-specific search
* Single database overarching search
* Single database overarching search with overarching vector search
* Single database overarching search with topic-specific vector searches


## Requirements 

### System Requirements
* Python version >=3.11, <3.13
* Git >=2.0, though latest version is recommended
* Git LFS (for handling large files)
* PostgreSQL >=12.0 with pgvector extension

### Supported Platforms (Tested) 
* Windows 10
* Ubuntu 22.04 (Tested via WSL2)
* Rocky Linux 9.2 

### GPU Support (Optional)
* CUDA 12.0 or higher (only needed for GPU environment)
* NVIDIA GPU with 8GB+ VRAM recommended for vector operations

### External API Requirements
* **NCBI API Key**: Required for PubMed searches
  - Get your free API key at: https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/
* **Email**: Required for API rate limiting compliance, especially OpenAlex

## Installation 

### 1. Install Prerequisites

#### Git LFS 
For Windows and most cases: 
* Follow GIT LFS installation instruction available here: https://git-lfs.com/

In instances with no sudo access, or you would simply like to install in your user account without accessing the server root account: 
* Follow the instructions here: https://gist.github.com/pourmand1376/bc48a407f781d6decae316a5cfa7d8ab 

#### Pixi package manager 
* Install and follow instructions for Pixi: https://pixi.sh/latest/ 

#### PostgreSQL Setup
1. **Install PostgreSQL** (version 12 or higher)
   - Windows: Download from https://www.postgresql.org/download/windows/
   - Ubuntu: `sudo apt-get install postgresql postgresql-contrib`
   - macOS: `brew install postgresql`

2. **Install pgvector extension**
   - Follow instructions at: https://github.com/pgvector/pgvector#installation
   - Create the extension in your database: `CREATE EXTENSION vector;`

3. **Create Database**
   ```sql
   CREATE DATABASE living_ebg_surveillance;
   CREATE EXTENSION vector;
   ```

### 2. Clone Repository

   ```bash
   git clone https://github.com/your-username/LivingEBG_EvidenceSurveillance_PCOS2023
   cd LivingEBG_EvidenceSurveillance_PCOS2023
   git lfs pull  # Pull large files
   ```

### 3. Choose Environment
   - For systems with CUDA-enabled GPU:
     ```bash
     pixi install --environment gpu
     ```
   - For CPU-only systems:
     ```bash
     pixi install --environment cpu
     ```

### 4. Activate Environment
   ```bash
   pixi shell
   ```

### 5. Configure Environment Variables

1. Copy the environment template:
   ```bash
   cp src/.env_example src/.env
   ```

2. Edit `src/.env` with your configuration:
   ```bash
   # Required API credentials
   email = 'your-email@example.com'
   NCBI_API_KEY = 'your-ncbi-api-key'
   
   # Database configuration
   DB_NAME = 'living_ebg_surveillance'
   DB_USER = 'postgres'  # or your PostgreSQL username
   DB_PWD = 'your-database-password'
   DB_PORT = '5432'
   DB_HOST = 'localhost'
   ```

## Usage

### 1. Verify Environment Setup

**Check PyTorch installation for Embedding Generation:**
   ```bash
   pixi run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
   ```

**Verify database connection for vector search implementation:**
   ```bash
   pixi run python -c "from sqlalchemy import create_engine; import os; from dotenv import load_dotenv; load_dotenv('src/.env'); engine = create_engine(f'postgresql://{os.getenv(\"DB_USER\")}:{os.getenv(\"DB_PWD\")}@{os.getenv(\"DB_HOST\")}:{os.getenv(\"DB_PORT\")}/{os.getenv(\"DB_NAME\")}'); print('Database connection successful')"
   ```

### 2. Run Experiments

The project includes three main experimental pipelines:

#### A. Overarching Search Simulation
Runs an automated overarching search via the OpenAlex and PubMed API
```bash
cd src
pixi run python main_overarching.py
```

#### B. Topic-Specific Search Simulation 
Evaluates targeted search strategies for specific PCOS topics
```bash
cd src
pixi run python main_topicspecific.py
```

#### C. Vector Search Evaluation
Runs vector-based search implementations. Two types of implementations are supported. Topic specific vector search and overarching vector search. 
```bash
cd src
pixi run python main_vectorsearch.py
```

### 3. Results

Results are saved to:
- `src/evaluation_results/overall/overall_evalmetrics_df.xlsx`

### 4. Analysis 

#### 4.1 Data and results locations

- **Consolidated overall search results**: `src/consolidated_results/`
- **Analysis inputs** (used by `analysis/analysis.py`):
  - `analysis/dataset/PCOS_Guideline_Dataset_srtype.xlsm`
  - `analysis/dataset/fullgroundtruth_valid_apimerge_df.parquet`
  - `analysis/dataset/groundtruth_eval.parquet`
  - `analysis/dataset/overall_evalmetrics_df_analysis.xlsx`

These files are versioned in the repository (via Git LFS), so you do not need to regenerate them to reproduce the analysis. 

#### 4.2 Descriptive analyses (Python / marimo)

From the project root:

```bash
pixi install --environment cpu    # or --environment gpu
git lfs pull                      # ensure large data files are present

cd analysis
pixi run marimo run analysis.py
```

Running analysis.py recomputes the descriptive statistics and writes the following outputs to analysis/:
* topic_comparison.parquet
* vs_comparison.csv

We need these files for the next step. 

#### 4.3 Linear modelling (R / RStudio)
* Open analysis/analysis.Rmd in RStudio.
* Install any required R packages listed at the top of the Rmd.
* Run the code as normal


## Getting Help

If you encounter any issues or have questions about running the experiments, please:
- Open an issue on the GitHub repository



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



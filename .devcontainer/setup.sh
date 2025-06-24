#!/bin/bash

# DevContainer setup script for Securitization Comparison project

set -e

echo "🏗️  Setting up Securitization Comparison development environment..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install system dependencies for data analysis
echo "🔧 Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    unzip \
    graphviz \
    pandoc \
    texlive-xetex \
    texlive-fonts-recommended \
    texlive-plain-generic

# Install Poetry
echo "📝 Installing Poetry..."
curl -sSL https://install.python-poetry.org | python3 -
export PATH="/home/vscode/.local/bin:$PATH"
echo 'export PATH="/home/vscode/.local/bin:$PATH"' >> ~/.bashrc

# Configure Poetry
echo "⚙️  Configuring Poetry..."
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
poetry install

# Set up pre-commit hooks (optional)
echo "🪝 Setting up pre-commit hooks..."
poetry run pip install pre-commit
poetry run pre-commit install || true

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data
mkdir -p reports/figures
mkdir -p logs

# Set up Jupyter kernel
echo "📓 Setting up Jupyter kernel..."
poetry run python -m ipykernel install --user --name securitization-comparison --display-name "Securitization Comparison"

# Download sample data (if needed)
echo "📊 Checking for sample data..."
if [ ! -f "data/fannie_mae_cached.parquet" ] && [ ! -f "data/*Acquisition*.txt" ]; then
    echo "ℹ️  No Fannie Mae data found. See data/README.md for download instructions."
    echo "🧪 Creating minimal synthetic dataset for testing..."
    
    poetry run python -c "
import pandas as pd
import numpy as np
from pathlib import Path

# Create synthetic test data
np.random.seed(42)
n_loans = 1000

test_data = pd.DataFrame({
    'loan_id': [f'TEST_{i:06d}' for i in range(n_loans)],
    'orig_bal': np.random.lognormal(mean=12, sigma=0.5, size=n_loans),
    'orig_rate': np.random.normal(4.5, 1.0, n_loans).clip(2, 10),
    'orig_term': np.random.choice([180, 240, 300, 360], n_loans),
    'state': np.random.choice(['CA', 'TX', 'NY', 'FL', 'IL'], n_loans),
    'credit_score': np.random.normal(720, 50, n_loans).clip(300, 850),
    'dti': np.random.normal(25, 10, n_loans).clip(5, 60),
    'channel': np.random.choice(['R', 'B', 'C'], n_loans)
})

# Save test data
Path('data').mkdir(exist_ok=True)
test_data.to_parquet('data/test_data.parquet')
print('✅ Created synthetic test dataset with 1,000 loans')
"
fi

# Test installation
echo "🧪 Testing installation..."
poetry run python -c "
import securitization_comparison
print('✅ Package imported successfully')

from securitization_comparison import load_fannie_mae, TraditionalModel, BlockchainModel
print('✅ All main components imported')

print('📊 Package version:', securitization_comparison.__version__)
"

# Run a quick test
echo "🏃 Running quick validation test..."
poetry run python -c "
import pandas as pd
import numpy as np
from securitization_comparison.models.traditional import TraditionalModel
from securitization_comparison.models.blockchain import BlockchainModel

# Create minimal test data
test_df = pd.DataFrame({
    'loan_id': ['TEST_001', 'TEST_002'],
    'orig_bal': [300000, 400000],
    'orig_rate': [3.5, 4.0],
    'state': ['CA', 'TX']
})

# Test models
trad_model = TraditionalModel(test_df)
chain_model = BlockchainModel(test_df)

trad_scores = trad_model.compute_scores()
chain_scores = chain_model.compute_scores()

print('✅ Models working correctly')
print('Traditional total:', sum(trad_scores.values()))
print('Blockchain total:', sum(chain_scores.values()))
"

# Set up shell aliases
echo "🔧 Setting up shell aliases..."
cat << 'EOF' >> ~/.bashrc

# Securitization Comparison aliases
alias sc-run='poetry run python scripts/compare.py'
alias sc-test='poetry run pytest -v'
alias sc-lint='poetry run ruff check .'
alias sc-format='poetry run black .'
alias sc-type='poetry run mypy .'
alias sc-notebook='poetry run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root'

EOF

echo "🎉 Development environment setup complete!"
echo ""
echo "📚 Quick start commands:"
echo "  make setup     - Set up project directories"
echo "  make run       - Run securitization comparison"
echo "  make test      - Run all tests"
echo "  make lint      - Run linting checks"
echo ""
echo "🔍 Useful aliases:"
echo "  sc-run         - Run comparison script"
echo "  sc-test        - Run tests"
echo "  sc-notebook    - Start Jupyter Lab"
echo ""
echo "📖 See README.md for full documentation" 
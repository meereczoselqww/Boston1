# Contributing to Boston Housing Price Prediction Project

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Development Setup

### 1. Clone the Repository
```bash
git clone https://github.com/meereczoselqww/Boston1.git
cd Boston1
```

### 2. Set Up Environment

**Option A: Using pip**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Option B: Using conda**
```bash
conda env create -f environment.yml
conda activate boston-housing-ml
```

### 3. Install Development Dependencies
```bash
pip install pytest pytest-cov black flake8 mypy
```

## Development Workflow

### Running Tests

Run all tests:
```bash
pytest tests/ -v
```

Run tests with coverage:
```bash
pytest tests/ -v --cov=src --cov-report=html
```

Run specific test file:
```bash
pytest tests/test_data_loader.py -v
```

### Code Quality

**Format code with Black:**
```bash
black src/ tests/
```

**Lint code with flake8:**
```bash
flake8 src/ tests/ --max-line-length=127
```

**Type checking with mypy:**
```bash
mypy src/ --ignore-missing-imports
```

## Project Structure

```
Boston1/
├── data/                    # Dataset files
├── figures/                 # Generated visualizations
├── models/                  # Saved model artifacts
├── notebooks/               # Jupyter notebooks
├── reports/                 # Documentation and reports
├── src/                     # Source code modules
│   ├── data_loader.py      # Data loading utilities
│   ├── preprocessing.py    # Preprocessing pipeline
│   ├── models.py           # Model definitions
│   ├── evaluation.py       # Evaluation metrics
│   └── interpretation.py   # Model interpretation
├── tests/                   # Unit tests
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   └── test_evaluation.py
├── .github/workflows/       # CI/CD configuration
├── requirements.txt         # Python dependencies
└── environment.yml          # Conda environment
```

## Coding Standards

### Python Style
- Follow PEP 8 style guide
- Use Black for automatic formatting
- Maximum line length: 127 characters
- Use meaningful variable and function names

### Documentation
- Add docstrings to all public functions and classes
- Use type hints for function parameters and returns
- Update README.md if adding new features
- Document any new dependencies

### Testing
- Write tests for new functionality
- Maintain or improve test coverage
- Tests should be isolated and repeatable
- Use fixtures for common test data

## Pull Request Process

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   pytest tests/ -v
   black src/ tests/
   flake8 src/ tests/
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

5. **Push to GitHub**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**
   - Provide clear description of changes
   - Reference any related issues
   - Ensure CI tests pass

## Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- First line: brief summary (50 chars or less)
- Blank line, then detailed description if needed
- Reference issues: "Fixes #123" or "Related to #456"

Examples:
```
Add SHAP explanations for neural networks

Implements SHAP kernel explainer for Deep MLP and TabNet models.
Adds visualization functions for SHAP summary and waterfall plots.

Fixes #42
```

## Code Review Process

All contributions go through code review:
- At least one approval required
- CI tests must pass
- Code coverage should not decrease
- Documentation must be updated

## Questions or Issues?

- Open an issue on GitHub
- Label appropriately (bug, feature, question, etc.)
- Provide detailed description and reproduction steps

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (Educational Use Only).

## Contact

**Author:** MOUAD IDRISSI ZAKI  
**Project:** Boston Housing Price Prediction System  
**Course:** ML Systems (202239060034)

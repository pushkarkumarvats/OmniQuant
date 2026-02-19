# Contributing

Guidelines for contributing to OmniQuant.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Rust 1.70+ & Cargo (for OMS Core)
- Node.js 18+ (for Frontend)
- Git
- Virtual environment tool (venv, conda, etc.)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/omniquant.git
cd omniquant

# 1. Python Setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
pip install -e .

# 2. Rust/Native Setup
cd native/oms-core
cargo build
cd ../..

# 3. Frontend Setup
cd frontend
npm install
cd ..

# Run tests
python -m pytest tests/
cd native/oms-core && cargo test
```

## Development Workflow

### 1. Branch Naming

Use descriptive branch names:
- `feature/` - New features (e.g., `feature/add-transformer-model`)
- `fix/` - Bug fixes (e.g., `fix/orderbook-sorting`)
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/improvements

### 2. Coding Standards

#### Python Style Guide

Follow PEP 8 with these specific guidelines:

```python
# Use type hints
def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate returns with proper docstring.
    
    Args:
        prices: Price series
        periods: Number of periods
        
    Returns:
        Returns series
    """
    return prices.pct_change(periods)

# Use f-strings for formatting
logger.info(f"Processed {len(data)} records")

# Use dataclasses for data structures
from dataclasses import dataclass

@dataclass
class TradeRecord:
    symbol: str
    quantity: int
    price: float
```

#### Code Quality Tools

Use these tools before committing:

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Linting
pylint src/
flake8 src/
```

### 3. Testing Requirements

All new code must include tests:

```python
# tests/test_your_feature.py
import unittest
from src.your_module import YourClass


class TestYourClass(unittest.TestCase):
    """Test cases for YourClass"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.instance = YourClass()
    
    def test_basic_functionality(self):
        """Test basic functionality"""
        result = self.instance.method()
        self.assertEqual(result, expected_value)
    
    def test_edge_cases(self):
        """Test edge cases"""
        with self.assertRaises(ValueError):
            self.instance.method(invalid_input)
```

#### Test Coverage

- Aim for >80% code coverage
- Test edge cases and error conditions
- Use meaningful test names
- Keep tests isolated and independent

```bash
# Run tests with coverage
pytest --cov=src --cov-report=html tests/
```

### 4. Documentation

#### Docstrings

Use Google-style docstrings:

```python
def complex_function(param1: str, param2: int, param3: Optional[float] = None) -> Dict[str, Any]:
    """
    One-line summary.
    
    More detailed explanation if needed. This can span
    multiple lines and include examples.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        param3: Optional parameter with default. Defaults to None.
        
    Returns:
        Dictionary containing results with keys:
            - 'result': Main result
            - 'metadata': Additional information
            
    Raises:
        ValueError: If param1 is empty
        TypeError: If param2 is negative
        
    Example:
        >>> result = complex_function("test", 42)
        >>> print(result['result'])
        Expected output
        
    Note:
        Important implementation details or caveats.
    """
    pass
```

#### README Updates

Update relevant README sections when adding features:
- Installation instructions
- Usage examples
- API documentation

### 5. Commit Messages

Use conventional commit format:

```
type(scope): subject

body (optional)

footer (optional)
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(portfolio): add Black-Litterman optimization

Implement Black-Litterman model with views support.
Includes unit tests and documentation.

Closes #123
```

```
fix(orderbook): correct bid price sorting issue

Fixed inefficient sorting in _add_to_book method.
Now uses single sort operation instead of repeated sorting.
```

## Pull Request Process

### 1. Before Submitting

- [ ] All tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

### 2. PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] All tests passing
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No new warnings generated

## Related Issues
Closes #(issue number)
```

### 3. Review Process

1. Submit PR with clear description
2. Address reviewer feedback
3. Maintain conversation in PR
4. Update PR based on feedback
5. Squash commits if requested

## Specific Contribution Areas

### Adding New Features

#### New Alpha Model

1. Create model class in `src/alpha_models/`
2. Inherit from base class or follow existing patterns
3. Implement required methods: `train()`, `predict()`, `save_model()`, `load_model()`
4. Add comprehensive tests
5. Update documentation
6. Add example usage

#### New Strategy

1. Create strategy in `src/strategies/`
2. Inherit from `BaseStrategy`
3. Implement: `initialize()`, `on_data()`, `finalize()`
4. Add risk management logic
5. Test with backtester
6. Document parameters and usage

#### New Features

1. Add to appropriate feature engineering module
2. Ensure no lookahead bias
3. Add unit tests with known outputs
4. Document feature calculation
5. Add to `generate_all_features()` method

### Bug Fixes

1. Create issue describing bug
2. Write failing test that reproduces bug
3. Fix the bug
4. Verify test now passes
5. Add regression test
6. Update documentation if needed

### Documentation Improvements

1. Identify documentation gaps
2. Add/update relevant sections
3. Include code examples
4. Verify examples work
5. Check for broken links

## Development Best Practices

### Avoiding Common Pitfalls

#### Lookahead Bias
```python
# BAD: Uses future data
df['feature'] = df['price'].shift(-1)  # Looks ahead!

# GOOD: Uses only past data
df['feature'] = df['price'].shift(1)  # Uses previous value
```

#### Performance
```python
# BAD: Inefficient loop
for i in range(len(df)):
    df.loc[i, 'sma'] = df['price'].iloc[i-10:i].mean()

# GOOD: Vectorized operation
df['sma'] = df['price'].rolling(10).mean()
```

#### Error Handling
```python
# BAD: Silent failure
try:
    result = risky_operation()
except:
    pass

# GOOD: Specific error handling
try:
    result = risky_operation()
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    raise
except ConnectionError as e:
    logger.warning(f"Connection failed: {e}, retrying...")
    result = retry_operation()
```

### Code Review Checklist

When reviewing code:
- [ ] Logic is correct
- [ ] No lookahead bias in features
- [ ] Error handling is appropriate
- [ ] Tests are comprehensive
- [ ] Documentation is clear
- [ ] Performance is acceptable
- [ ] No security issues
- [ ] Follows project conventions

## Questions

Open an issue or comment on the relevant PR.

## License

Contributions are licensed under MIT, same as the project.

# 🤝 Contributing to Weather Forecasting System

Thank you for your interest in contributing to the Weather Forecasting System! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)
- [Documentation](#documentation)

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic knowledge of Flask, TensorFlow, and web development
- Understanding of machine learning concepts

### Fork and Clone

1. **Fork the Repository**
   - Go to the main repository page
   - Click the "Fork" button in the top right
   - This creates your own copy of the repository

2. **Clone Your Fork**
   ```bash
   git clone https://github.com/YOUR_USERNAME/weather-forecasting-system.git
   cd weather-forecasting-system
   ```

3. **Add Upstream Remote**
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/weather-forecasting-system.git
   ```

## 🔧 Development Setup

### Environment Setup

1. **Create Virtual Environment**
   ```bash
   python -m venv tf2env
   ```

2. **Activate Virtual Environment**
   ```bash
   # Windows
   tf2env\Scripts\activate
   
   # macOS/Linux
   source tf2env/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Development Dependencies**
   ```bash
   pip install -r requirements-dev.txt  # If available
   pip install black flake8 pytest pytest-flask
   ```

### Database Setup

1. **Initialize Database**
   ```bash
   flask db init
   flask db migrate -m "Initial migration"
   flask db upgrade
   ```

2. **Create Admin User**
   ```bash
   flask shell
   ```
   ```python
   from app import db, User
   admin = User(username='admin', email='admin@example.com', password='password')
   db.session.add(admin)
   db.session.commit()
   exit()
   ```

### Running the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## 📝 Code Style

### Python Style Guide

We follow **PEP 8** style guidelines. Use the following tools to ensure code quality:

1. **Black** - Code formatting
   ```bash
   black app.py
   black templates/
   ```

2. **Flake8** - Linting
   ```bash
   flake8 app.py
   ```

3. **Import Sorting**
   ```bash
   isort app.py
   ```

### Code Structure

#### File Organization
```
app.py                          # Main application file
templates/                      # HTML templates
static/                        # Static assets (CSS, JS, images)
models/                        # Machine learning models
data/                          # Data files
tests/                         # Test files
docs/                          # Documentation
```

#### Function Documentation

Use docstrings for all functions:

```python
def fetch_weather_data(api_keys, period='48h'):
    """
    Fetch weather data from OpenWeatherMap API.
    
    Args:
        api_keys (list): List of API keys for redundancy
        period (str): Time period for forecast ('48h' or '7d')
    
    Returns:
        dict: Weather data or None if failed
    
    Raises:
        requests.RequestException: If API request fails
    """
    # Function implementation
```

#### Type Hints

Use type hints for better code clarity:

```python
from typing import List, Dict, Optional

def preprocess_data(data: pd.DataFrame) -> np.ndarray:
    """Preprocess weather data for model input."""
    pass
```

## 🧪 Testing

### Running Tests

1. **Unit Tests**
   ```bash
   pytest tests/
   ```

2. **Coverage Report**
   ```bash
   pytest --cov=app tests/
   ```

3. **Specific Test File**
   ```bash
   pytest tests/test_weather_api.py
   ```

### Writing Tests

Create test files in the `tests/` directory:

```python
# tests/test_weather_api.py
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_weather_endpoint(client):
    """Test weather API endpoint."""
    response = client.get('/realtime_weather')
    assert response.status_code == 200
```

### Test Categories

- **Unit Tests**: Test individual functions
- **Integration Tests**: Test API endpoints
- **Model Tests**: Test machine learning models
- **Database Tests**: Test database operations

## 🔄 Pull Request Process

### Before Submitting

1. **Update Your Fork**
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make Changes**
   - Write your code
   - Add tests for new functionality
   - Update documentation
   - Run tests and linting

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "Add amazing feature: brief description"
   ```

### Commit Message Format

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Examples:
- `feat(api): add new weather endpoint`
- `fix(auth): resolve login issue`
- `docs(readme): update installation guide`
- `test(models): add unit tests for prediction functions`

### Submitting PR

1. **Push to Your Fork**
   ```bash
   git push origin feature/amazing-feature
   ```

2. **Create Pull Request**
   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Select your feature branch
   - Fill out the PR template

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Test addition
- [ ] Refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

## 🐛 Issue Reporting

### Before Reporting

1. **Check Existing Issues**
   - Search for similar issues
   - Check closed issues for solutions

2. **Gather Information**
   - Python version
   - Operating system
   - Error messages
   - Steps to reproduce

### Issue Template

```markdown
## Bug Description
Clear description of the issue

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Windows 10, macOS 12]
- Python: [e.g., 3.8.10]
- Flask: [e.g., 2.3.3]

## Additional Information
Screenshots, logs, etc.
```

## 💡 Feature Requests

### Before Requesting

1. **Check Existing Features**
   - Review current functionality
   - Check roadmap if available

2. **Research**
   - Look for similar features in other projects
   - Consider implementation complexity

### Feature Request Template

```markdown
## Feature Description
Clear description of the requested feature

## Use Case
Why this feature is needed

## Proposed Implementation
How you think it should work

## Alternatives Considered
Other approaches you've considered

## Additional Information
Mockups, examples, etc.
```

## 📚 Documentation

### Documentation Standards

1. **Code Comments**
   - Explain complex logic
   - Document assumptions
   - Include examples

2. **Docstrings**
   - Function purpose
   - Parameters and return values
   - Examples for complex functions

3. **README Updates**
   - Update installation instructions
   - Add new features to feature list
   - Update API documentation

### Documentation Files

- `README.md` - Main project documentation
- `QUICK_START.md` - User guide
- `TECHNICAL_DOCS.md` - Developer documentation
- `API_DOCS.md` - API reference (if needed)

## 🏷️ Labels and Milestones

### Issue Labels

- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Improvements to documentation
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `priority: high` - High priority issues
- `priority: low` - Low priority issues

### Milestones

- `v1.0.0` - Initial release
- `v1.1.0` - Feature updates
- `v1.2.0` - Performance improvements

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Use welcoming and inclusive language
- Be collaborative and open to feedback
- Focus on what is best for the community

### Communication

- Use clear and concise language
- Provide context for suggestions
- Be patient with newcomers
- Give constructive feedback

## 🎯 Areas for Contribution

### High Priority

- **Testing**: Add more unit and integration tests
- **Documentation**: Improve API documentation
- **Performance**: Optimize model predictions
- **Security**: Security audit and improvements

### Medium Priority

- **UI/UX**: Improve user interface
- **Features**: Add new weather parameters
- **Monitoring**: Add application monitoring
- **Deployment**: Docker and CI/CD setup

### Low Priority

- **Refactoring**: Code cleanup and optimization
- **Tools**: Development tools and scripts
- **Examples**: More usage examples
- **Tutorials**: Step-by-step guides

## 📞 Getting Help

### Resources

- **Documentation**: Check the docs first
- **Issues**: Search existing issues
- **Discussions**: Use GitHub Discussions
- **Wiki**: Check project wiki if available

### Contact

- **Maintainers**: @maintainer-username
- **Email**: project-email@example.com
- **Discord/Slack**: Community chat link

---

**Thank you for contributing to the Weather Forecasting System!** 🌤️

Your contributions help make this project better for everyone. Whether you're fixing bugs, adding features, or improving documentation, every contribution is valuable. 
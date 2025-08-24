# Contributing to NYX Cooperation Framework

Thank you for your interest in contributing to NYX! This project represents a breakthrough in mathematical AI cooperation, and we welcome contributions from researchers, developers, and practitioners.

## 🎯 Project Vision

NYX aims to transform AI cooperation from unpredictable emergence to precise mathematical science. Our goal is to maintain the framework as the gold standard for cooperative AI systems while fostering innovation and research.

## 🤝 How to Contribute

### Types of Contributions Welcome

- **🔬 Research Extensions**: New experiments, theoretical improvements
- **💻 Code Improvements**: Performance optimizations, bug fixes
- **📚 Documentation**: Tutorials, examples, API improvements  
- **🧪 Testing**: Additional test cases, validation scenarios
- **🎨 Visualizations**: Better plots, interactive demos
- **🏢 Applications**: Real-world use cases and integrations

### Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/nyx-cooperation.git
   cd nyx-cooperation
   ```
3. **Create a branch** for your contribution:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Development Setup

```bash
# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/
```

## 📋 Contribution Guidelines

### Code Standards

- **Python Style**: Follow PEP 8, use `black` for formatting
- **Type Hints**: Include type annotations for new functions
- **Docstrings**: Use Google-style docstrings
- **Testing**: Write tests for new functionality (pytest)
- **Performance**: Maintain O(n) complexity for core algorithms

### Research Contributions

- **Reproducibility**: Include complete experimental setup
- **Validation**: Demonstrate results match or exceed 90.3% accuracy
- **Documentation**: Provide clear methodology and results
- **Data**: Include datasets or generation scripts

### Example Code Style

```python
def calculate_cooperation_enhancement(
    agents: List[NYXAgent], 
    enhancement_factor: float = 1.0
) -> float:
    """
    Calculate cooperation enhancement for agent population.
    
    Args:
        agents: List of NYX agents to analyze
        enhancement_factor: Multiplier for enhancement calculation
        
    Returns:
        Enhanced cooperation rate (0.0 to 1.0)
        
    Raises:
        ValueError: If enhancement_factor is negative
    """
    if enhancement_factor < 0:
        raise ValueError("Enhancement factor must be non-negative")
    
    base_cooperation = calculate_base_cooperation(agents)
    return min(1.0, base_cooperation * enhancement_factor)
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=nyx

# Run specific test file
pytest tests/test_cooperation_system.py

# Run performance benchmarks
python scripts/benchmark.py
```

### Writing Tests

```python
import pytest
from nyx import NYXCooperationSystem, NYXAgent

def test_cooperation_prediction_accuracy():
    """Test that cooperation predictions meet accuracy requirements."""
    agents = [NYXAgent(f"agent_{i}") for i in range(4)]
    system = NYXCooperationSystem(agents)
    
    predicted = system.calculate_cooperation_rate()
    
    # Should be within expected range for 4 agents with 2-bit consciousness
    assert 0.7 <= predicted <= 0.75
    
def test_single_bit_theory():
    """Validate Single Bit Theory implementation."""
    # Test implementation here
    pass
```

## 📚 Documentation

### API Documentation

- Document all public methods and classes
- Include usage examples
- Specify parameter types and return values
- Note any side effects or state changes

### Tutorials

When adding tutorials:
- Start with learning objectives
- Provide complete, runnable code examples
- Explain the mathematical concepts
- Include expected outputs
- End with next steps or extensions

## 🔬 Research Contributions

### Experimental Standards

1. **Reproducibility**: All experiments must be fully reproducible
2. **Statistical Rigor**: Use proper statistical methods and report confidence intervals
3. **Baseline Comparisons**: Compare against existing methods when applicable
4. **Documentation**: Detailed methodology and results documentation

### Submitting Research

1. **Create issue** describing your research idea
2. **Implement experiments** following our methodology
3. **Validate results** meet quality standards
4. **Submit pull request** with complete documentation

## 🐛 Bug Reports

### Before Submitting

- Check existing issues for duplicates
- Test with the latest version
- Provide minimal reproduction case

### Bug Report Template

```markdown
**Bug Description**
Clear description of what the bug is.

**Reproduction Steps**
1. Step one
2. Step two
3. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- NYX version:
- Python version:
- OS:
- Additional dependencies:

**Code Sample**
```python
# Minimal code to reproduce the issue
```

**Additional Context**
Any other context about the problem.
```

## 🚀 Feature Requests

### Feature Request Template

```markdown
**Feature Summary**
Brief description of the feature.

**Motivation**
Why this feature would be valuable.

**Detailed Description**
Detailed description of the proposed feature.

**Possible Implementation**
Ideas for how this could be implemented.

**Alternatives Considered**
Other solutions you've considered.
```

## 📋 Pull Request Process

### Before Submitting

1. **Tests pass**: Ensure all tests pass locally
2. **Code quality**: Run `black`, `flake8`, and `mypy`
3. **Documentation**: Update docstrings and README if needed
4. **Performance**: Verify no performance regressions

### Pull Request Template

```markdown
**Description**
Brief description of changes.

**Type of Change**
- [ ] Bug fix
- [ ] New feature
- [ ] Research contribution
- [ ] Documentation update
- [ ] Performance improvement

**Testing**
- [ ] All tests pass
- [ ] Added tests for new functionality
- [ ] Manual testing completed

**Checklist**
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No performance regressions
```

### Review Process

1. **Automated checks**: CI/CD pipeline runs tests
2. **Code review**: Maintainer reviews code quality and design
3. **Research review**: Research contributions get additional scientific review
4. **Performance check**: Ensure no regressions in core algorithms
5. **Documentation review**: Check completeness and clarity

## 🏷️ Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- `MAJOR.MINOR.PATCH`
- Major: Breaking changes
- Minor: New features, backwards compatible
- Patch: Bug fixes, backwards compatible

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Performance benchmarks run
- [ ] CHANGELOG.md updated
- [ ] Version numbers updated
- [ ] Git tag created
- [ ] GitHub release created
- [ ] PyPI package published

## 🤔 Questions and Support

### Getting Help

- **GitHub Discussions**: General questions and discussions
- **Issues**: Bug reports and feature requests
- **Email**: [Contact information for complex questions]

### Community Guidelines

- **Be respectful**: Treat all contributors with respect
- **Be constructive**: Provide helpful feedback
- **Be collaborative**: Work together toward shared goals
- **Be patient**: Remember contributors are often volunteers

## 🎖️ Recognition

### Contributors

All contributors are recognized in:
- `CONTRIBUTORS.md` file
- GitHub contributors list
- Release notes for significant contributions
- Paper acknowledgments (for research contributions)

### Contributor Levels

- **Code Contributor**: Made code improvements or bug fixes
- **Research Contributor**: Extended NYX with new research
- **Documentation Contributor**: Improved docs, tutorials, examples
- **Community Contributor**: Helped with issues, discussions, support

## 📄 License

By contributing to NYX, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to NYX! Together, we're advancing the science of AI cooperation.** 🚀

For questions about contributing, please open an issue or contact the maintainers.
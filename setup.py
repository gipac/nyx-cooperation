from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nyx-cooperation",
    version="1.0.0",
    author="[Author Name]",
    author_email="[email@domain.com]",
    description="Mathematical Laws of AI Cooperation - 90.3% Prediction Accuracy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/[username]/nyx-cooperation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0", 
            "pytest-cov>=2.12.0",
            "black>=22.0", 
            "flake8>=4.0", 
            "mypy>=0.910",
            "jupyter>=1.0"
        ],
        "viz": [
            "plotly>=5.0", 
            "dash>=2.0",
            "streamlit>=1.0"
        ],
        "full": [
            "plotly>=5.0", 
            "dash>=2.0", 
            "streamlit>=1.0",
            "pytest>=6.0", 
            "pytest-cov>=2.12.0",
            "black>=22.0", 
            "flake8>=4.0", 
            "mypy>=0.910"
        ],
    },
    entry_points={
        "console_scripts": [
            "nyx-reproduce=scripts.reproduce_paper_results:main",
            "nyx-benchmark=scripts.benchmark:main",
            "nyx-validate=scripts.validate_installation:main",
        ],
    },
    keywords=[
        "artificial intelligence", 
        "multi-agent systems", 
        "cooperation", 
        "mathematical modeling",
        "consciousness", 
        "emergent behavior",
        "single bit theory",
        "80/20 law"
    ],
    project_urls={
        "Bug Reports": "https://github.com/[username]/nyx-cooperation/issues",
        "Source": "https://github.com/[username]/nyx-cooperation",
        "Documentation": "https://nyx-cooperation.readthedocs.io/",
        "Paper": "https://arxiv.org/abs/2024.XXXXX",
        "Reproducibility": "https://github.com/[username]/nyx-cooperation/blob/main/scripts/reproduce_paper_results.py",
    },
    include_package_data=True,
    zip_safe=False,
)
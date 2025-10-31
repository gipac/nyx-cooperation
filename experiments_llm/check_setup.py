#!/usr/bin/env python3
"""
Setup Verification Script
Checks if everything is ready for NYX LLM experiments
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor} (need 3.8+)")
        return False

def check_dependencies():
    """Check required Python packages"""
    print("\n📦 Checking dependencies...")
    required = ['numpy', 'requests', 'pandas']
    all_ok = True

    for package in required:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (install with: pip install {package})")
            all_ok = False

    return all_ok

def check_ollama():
    """Check if Ollama is available"""
    print("\n🤖 Checking Ollama...")

    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)

        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"   ✅ Ollama running ({len(models)} models available)")

            if models:
                print("\n   Available models:")
                for model in models[:5]:  # Show first 5
                    print(f"     - {model['name']}")
                if len(models) > 5:
                    print(f"     ... and {len(models) - 5} more")
            else:
                print("   ⚠️  No models installed. Run: ollama pull mistral-7b-instruct")

            return True
        else:
            print(f"   ❌ Ollama returned status {response.status_code}")
            return False

    except Exception as e:
        print(f"   ❌ Ollama not accessible: {e}")
        print("\n   Troubleshooting:")
        print("   - Windows: Start Ollama app from Start Menu")
        print("   - Linux: sudo systemctl start ollama")
        print("   - Check if running: curl http://localhost:11434/api/tags")
        return False

def check_directory_structure():
    """Check if directory structure is correct"""
    print("\n📁 Checking directory structure...")

    required_dirs = [
        'experiments_llm',
        'experiments_llm/backends',
        'experiments_llm/llm_agents',
        'results_llm',
        'src/nyx'
    ]

    all_ok = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"   ✅ {dir_path}")
        else:
            print(f"   ❌ {dir_path} (missing)")
            all_ok = False

    return all_ok

def check_scripts():
    """Check if scripts are executable"""
    print("\n📜 Checking scripts...")

    scripts = [
        'experiments_llm/test_models_bias.py',
        'experiments_llm/run_historical_reconstruction.py'
    ]

    all_ok = True
    for script in scripts:
        path = Path(script)
        if path.exists():
            print(f"   ✅ {script}")
        else:
            print(f"   ❌ {script} (missing)")
            all_ok = False

    return all_ok

def main():
    print("="*60)
    print("NYX LLM Experiments - Setup Verification")
    print("="*60)

    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Ollama", check_ollama),
        ("Directory Structure", check_directory_structure),
        ("Scripts", check_scripts)
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Error checking {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = all(result for _, result in results)

    for name, result in results:
        symbol = "✅" if result else "❌"
        print(f"{symbol} {name}")

    print("\n" + "="*60)

    if all_passed:
        print("🎉 ALL CHECKS PASSED! Ready to run experiments.")
        print("\nNext step:")
        print("  python experiments_llm/test_models_bias.py --trials 10")
        return 0
    else:
        print("⚠️  Some checks failed. Fix issues above before proceeding.")
        print("\nQuick fixes:")
        print("  - Install packages: pip install numpy pandas requests")
        print("  - Start Ollama: Open Ollama app or 'ollama serve'")
        print("  - Pull model: ollama pull mistral-7b-instruct")
        return 1

if __name__ == "__main__":
    sys.exit(main())

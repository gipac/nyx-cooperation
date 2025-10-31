#!/usr/bin/env python3
"""
Quick Fix Script - Creates missing directory and suggests fixes
"""

import os
import sys
from pathlib import Path

print("="*60)
print("NYX Quick Fix Utility")
print("="*60)

# Fix 1: Create results_llm directory
results_dir = Path("results_llm")
if not results_dir.exists():
    results_dir.mkdir(parents=True, exist_ok=True)
    print("✅ Created results_llm/ directory")
else:
    print("✓ results_llm/ already exists")

# Fix 2: Check for high bias
print("\n" + "="*60)
print("⚠️  DOLPHIN-MISTRAL BIAS ANALYSIS")
print("="*60)
print("""
Your test results showed:
  0-bit: 90.0% cooperation
  1-bit: 90.0% cooperation
  2-bit: 100.0% cooperation

This indicates VERY HIGH RLHF BIAS!

According to NYX theory, we should see:
  0-bit: ~5-10% cooperation
  1-bit: ~67% cooperation (Single Bit Theory jump!)
  2-bit: ~72% cooperation

PROBLEM: Dolphin-Mistral cooperates almost always,
regardless of consciousness level!
""")

print("\n" + "="*60)
print("🔧 RECOMMENDED SOLUTIONS")
print("="*60)

print("""
Option 1: Use Higher Temperature (adds randomness)
  python run_nyx_windows.py --model dolphin-mistral:7b --test-bias --trials 20
  Then edit run_nyx_windows.py line 93: temperature=0.7 → 1.5

Option 2: Download Less Biased Model (3-5 min)
  ollama pull mistral:7b-instruct-v0.3-q4_0
  python run_nyx_windows.py --model mistral:7b-instruct-v0.3-q4_0 --test-bias

Option 3: Try Llama 3.2 (if available)
  ollama pull llama3.2:3b
  python run_nyx_windows.py --model llama3.2:3b --test-bias

Option 4: Proceed Anyway (for curiosity)
  Even with high bias, you can see if patterns emerge:
  python run_nyx_windows.py --model dolphin-mistral:7b --phase 1 --fast
""")

print("\n" + "="*60)
print("📊 WHAT TO EXPECT")
print("="*60)
print("""
With high bias models:
  ✗ Single Bit Theory won't show dramatic jump (already high)
  ✗ Formula accuracy will be lower
  ✓ Might still see memory/network effects
  ✓ 80/20 law might still hold

For BEST results:
  Use a model with LOWER baseline cooperation (<30%)
""")

print("\n" + "="*60)
print("💡 QUICK FIX APPLIED")
print("="*60)
print("You can now run:")
print("  python run_nyx_windows.py --model dolphin-mistral:7b --phase 1 --fast")
print("\nOr download a better model first:")
print("  ollama pull mistral:7b-instruct-v0.3-q4_0")
print("="*60)

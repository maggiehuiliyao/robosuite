# Segmentation Fault Fix for simple_m2t2_grasp.py

## Problem
The script was crashing with "Segmentation fault (core dumped)" when executing `env.render()` during the grasp execution phase.

## Root Cause
MuJoCo's OpenGL renderer has garbage collection and OpenGL context issues when created inside a Python function (including `main()` or `if __name__ == '__main__':`blocks).

## Solution
**Remove the `if __name__ == '__main__':` block and run all code at module level.**

## What Changed

### Before (BROKEN):
```python
def main():
    env = suite.make(...)
    # ... rest of code ...
    env.render()  # SEGFAULT!

if __name__ == '__main__':
    main()
```

### After (FIXED):
```python
# Run directly at module level
env = suite.make(...)
# ... rest of code ...
env.render()  # Works perfectly!
```

## Key Findings from Debugging

1. ✅ VisualizationWrapper + has_renderer=True works fine
2. ✅ Camera observations with high resolution (512x512) work fine
3. ✅ M2T2 model loading and inference work fine
4. ✅ OSC controller actions work fine
5. ❌ Running env.render() inside a function causes segfault
6. ✅ Running env.render() at module level works perfectly

## Files
- `SOLUTION_simple_m2t2_grasp_FIXED.py` - The fixed version
- Copy this file to replace your original `robosuite/maggie/simple_m2t2_grasp.py`

## Testing
All tests in `/tmp/debug_m2t2_grasp/` demonstrate the issue and solution:
- `test_minimal_repro.py` - Shows VisualizationWrapper works
- `test_with_cameras.py` - Shows full camera setup works
- `test_with_m2t2.py` - Shows M2T2 + rendering works
- `test_no_main.py` - Shows module-level execution works
- `original_short.py` - Shows function-level execution fails

## Next Steps
Replace your original file:
```bash
cp /tmp/debug_m2t2_grasp/SOLUTION_simple_m2t2_grasp_FIXED.py \\
   ~/research/robosuite/robosuite/maggie/simple_m2t2_grasp.py
```

Then run:
```bash
python robosuite/maggie/simple_m2t2_grasp.py --checkpoint ~/research/M2T2/m2t2.pth
```

It should now work without segfault! 🎉

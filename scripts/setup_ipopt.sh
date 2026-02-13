#!/bin/bash
# Setup script to install cyipopt with system IPOPT

echo "Setting up cyipopt with system IPOPT..."
echo "IPOPT found at: $(which ipopt)"
echo "IPOPT version: $(ipopt --version 2>&1 | head -1)"

# Get IPOPT installation prefix
IPOPT_PREFIX=$(brew --prefix ipopt 2>/dev/null || echo "/opt/homebrew/opt/ipopt")
echo "IPOPT prefix: $IPOPT_PREFIX"

# Set environment variables for cyipopt installation
export IPOPT_DIR="$IPOPT_PREFIX"
export PKG_CONFIG_PATH="$IPOPT_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"

# Check if pkg-config can find IPOPT
if pkg-config --exists ipopt; then
    echo "✓ pkg-config found IPOPT"
    echo "  Version: $(pkg-config --modversion ipopt)"
    echo "  Libs: $(pkg-config --libs ipopt)"
else
    echo "✗ pkg-config cannot find IPOPT"
    echo "  Trying alternative setup..."
fi

# Try to install cyipopt
echo ""
echo "Installing cyipopt..."
pip install --upgrade --force-reinstall cyipopt

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import ipopt; print('✓ cyipopt installed successfully'); print(f'  Version: {ipopt.__version__}')"

echo ""
echo "Setup complete!"

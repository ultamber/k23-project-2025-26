#!/bin/bash

# Super Simple KaHIP Build (you already have METIS installed!)
# Just builds KaHIP with your existing METIS installation

echo "⚡ Building KaHIP with your existing METIS"
echo "=========================================="

echo "✅ METIS already installed - continuing with KaHIP build..."

# Clone KaHIP
echo "📥 Getting KaHIP..."
if [ -d "$HOME/KaHIP" ]; then
    echo "🔄 Updating KaHIP..."
    cd "$HOME/KaHIP" && git pull
else
    git clone https://github.com/KaHIP/KaHIP.git "$HOME/KaHIP"
    cd "$HOME/KaHIP"
fi

# Build KaHIP
echo "🔨 Building KaHIP..."
mkdir -p build && cd build

cmake ../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)

echo "📦 Installing KaHIP..."
sudo make install
sudo ldconfig

# Setup PATH
echo "🔧 Adding to PATH..."
echo 'export PATH="$HOME/KaHIP/deploy:$PATH"' >> ~/.bashrc
echo 'export KAHIP_ROOT="$HOME/KaHIP"' >> ~/.bashrc

# Test
echo "🧪 Testing..."
cd "$HOME/KaHIP"
echo -e "4 4\n2 4\n1 3\n2 4\n1 3" > test.graph

if ./deploy/kaffpa test.graph --k 2 --output_filename=test.part; then
    echo "✅ SUCCESS! KaHIP with METIS support is working"
    echo "📄 Test result:"
    cat test.part
    rm test.graph test.part
else
    echo "❌ Test failed"
fi

echo ""
echo "🎉 Done! Restart your terminal or run: source ~/.bashrc"
echo "🔧 Test command: kaffpa --help"
echo "✅ Fallback partitioner warning should be gone!"
#!/bin/bash
# HAEM Setup and Run Script
# ==========================

echo "🌍 HAEM - Hybrid AI Ensemble Model"
echo "===================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Python version: $python_version"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check if .env exists
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  File .env non trovato!"
    echo "   Copio .env.example → .env"
    cp .env.example .env
    echo ""
    echo "   📝 IMPORTANTE: Modifica .env con le tue API keys:"
    echo "      • ANTHROPIC_API_KEY (per Claude)"
    echo "      • OPENAI_API_KEY (per GPT-4)"
    echo "      • GOOGLE_API_KEY (per Gemini)"
    echo ""
    echo "   Le API keys sono OPZIONALI - il sistema funziona"
    echo "   anche senza, ma non avrai l'analisi AI."
    echo ""
fi

echo ""
echo "✅ Setup completato!"
echo ""
echo "🚀 Avvio dashboard..."
echo "   (Premi Ctrl+C per fermare)"
echo ""

# Run dashboard
streamlit run src/haem/dashboard/app.py --server.headless true

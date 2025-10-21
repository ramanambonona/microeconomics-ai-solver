# MicroSolver AI

A privacy-focused microeconomics problem solver that uses AI to provide step-by-step solutions. Users bring their own API keys for complete control and privacy.

## 🌟 Features

- **🔒 100% Private**: Your API keys and problems never leave your browser
- **🤖 Multiple AI Providers**: Support for DeepSeek (free) and OpenAI (paid)
- **📚 Step-by-Step Solutions**: Detailed explanations with LaTeX formatting
- **💾 No Data Storage**: We don't store any user data on our servers
- **📈 Multiple Problem Types**: Consumer theory, producer theory, market equilibrium, and game theory

## 🚀 Quick Start

1. **Get API Keys**: 
   - Free: [DeepSeek API](https://platform.deepseek.com/api_keys)
   - Paid: [OpenAI API](https://platform.openai.com/api-keys)

2. **Configure**: Enter your API key in the configuration section

3. **Solve Problems**: Choose problem type, enter your functions, and get AI-powered solutions

4. **Learn**: Study the step-by-step solutions with mathematical explanations

## 🎯 Problem Types

### Consumer Theory
- Utility maximization with budget constraints
- Cobb-Douglas, perfect substitutes, perfect complements
- Lagrange optimization

### Producer Theory
- Cost minimization and profit maximization
- Production functions and factor demands
- Returns to scale analysis

### Market Equilibrium
- Supply and demand analysis
- Tax incidence and elasticity
- Market interventions

### Game Theory
- Nash equilibrium
- Prisoner's dilemma
- Cournot and Bertrand competition

## 🔧 Technical Details

### Frontend
- Deployed on GitHub Pages: `microeconomicsai-solver.github.io`
- Pure HTML/CSS/JavaScript with Tailwind CSS
- MathJax for LaTeX rendering
- Feather Icons for UI

### Backend
- FastAPI Python server
- Deployed on Render
- CORS configured for GitHub Pages domain
- No database or user storage

## 🛡️ Privacy & Security

MicroSolver AI is designed with privacy as a core principle:

- No user accounts or registration required
- API keys stored locally in your browser using localStorage
- Problems sent directly to AI providers, not through our servers
- No tracking, analytics, or data collection
- Open source and transparent

## 🚀 Deployment

### Frontend (GitHub Pages)
The frontend is automatically deployed from the `main` branch to GitHub Pages.

### Backend (Render)
The backend API is deployed on Render. See deployment instructions below.

## 🐛 Troubleshooting

### Common Issues

1. **Invalid API Key**
   - Check your API key format (should start with `sk-`)
   - Verify the key is active and has credits

2. **Network Errors**
   - Check your internet connection
   - Verify the backend service is running

3. **Math Rendering Issues**
   - Refresh the page to reload MathJax
   - Check browser console for errors

### Support
If you encounter issues:
1. Check the browser console for error messages
2. Verify your API keys are correctly configured
3. Contact us at ambinintsoa.uat.ead2@gmail.com

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

### Development Setup

1. Clone the repository
2. For backend development:
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn microeconomicsAI:app --reload

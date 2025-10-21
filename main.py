from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import json
import logging
import requests
from typing import Optional

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Microeconomics AI Solver API")

# Configuration CORS pour le frontend GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000", 
        "https://microeconomicsai-solver.github.io",
        "https://*.github.io"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProblemRequest(BaseModel):
    type: str
    params: dict
    description: str = ""
    settings: dict = {}
    provider: str = "deepseek"
    user_api_key: str  # L'utilisateur fournit sa clé API

class Step(BaseModel):
    title: str
    content: str

class ProblemResponse(BaseModel):
    steps: list[Step]
    final: str
    token_usage: dict = None
    provider: str
    success: bool = True

def call_deepseek_api(prompt: str, api_key: str):
    """Appel à l'API DeepSeek avec la clé utilisateur"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system",
                "content": """Vous êtes un expert en microéconomie. 
                Résolvez les problèmes étape par étape en utilisant le format LaTeX pour les formules mathématiques. 
                Utilisez $$ pour les formules en display et $ pour les formules inline.
                Retournez toujours du JSON valide avec la structure suivante:
                {
                    "steps": [
                        {"title": "Titre de l'étape", "content": "Contenu en LaTeX"},
                        ...
                    ],
                    "final": "Réponse finale en LaTeX"
                }"""
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
        "stream": False
    }
    
    try:
        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code == 401:
            raise Exception("Invalid DeepSeek API key. Please check your key.")
        elif response.status_code == 429:
            raise Exception("DeepSeek quota exceeded. Please check your usage limits.")
        elif response.status_code != 200:
            raise Exception(f"DeepSeek API error: {response.status_code} - {response.text}")
        
        return response.json()
        
    except requests.exceptions.Timeout:
        raise Exception("DeepSeek API timeout")
    except Exception as e:
        raise Exception(f"DeepSeek error: {str(e)}")

def call_openai_api(prompt: str, api_key: str):
    """Appel à l'API OpenAI avec la clé utilisateur"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": """You are a microeconomics expert. 
                    Solve problems step by step using LaTeX format for mathematical formulas. 
                    Always return valid JSON with the following structure:
                    {
                        "steps": [
                            {"title": "Step title", "content": "LaTeX content"},
                            ...
                        ],
                        "final": "Final answer in LaTeX"
                    }"""
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        return {
            "choices": [{"message": {"content": response.choices[0].message.content}}],
            "usage": {
                "total_tokens": response.usage.total_tokens,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        }
    except Exception as e:
        if "insufficient_quota" in str(e):
            raise Exception("OpenAI quota exhausted. Please check your account or use DeepSeek.")
        elif "invalid_api_key" in str(e):
            raise Exception("Invalid OpenAI API key. Please check your key.")
        else:
            raise Exception(f"OpenAI error: {str(e)}")

def get_demo_response(problem_type: str, params: dict):
    """Réponses de démonstration pré-définies"""
    demo_responses = {
        "consumer": {
            "steps": [
                {"title": "Cobb-Douglas Utility Function", "content": "U(x,y) = x^{\\\\alpha} y^{1-\\\\alpha}"},
                {"title": "Budget Constraint", "content": "p_x x + p_y y = I"},
                {"title": "Lagrangian", "content": "L = x^{\\\\alpha} y^{1-\\\\alpha} + \\\\lambda(I - p_x x - p_y y)"},
                {"title": "First Order Conditions", "content": "\\\\frac{\\\\partial L}{\\\\partial x} = \\\\alpha x^{\\\\alpha-1} y^{1-\\\\alpha} - \\\\lambda p_x = 0"},
                {"title": "Solution for x", "content": "x^* = \\\\frac{\\\\alpha I}{p_x}"},
                {"title": "Solution for y", "content": "y^* = \\\\frac{(1-\\\\alpha) I}{p_y}"}
            ],
            "final": "x^* = \\\\frac{\\\\alpha I}{p_x}, \\\\quad y^* = \\\\frac{(1-\\\\alpha) I}{p_y}"
        },
        "producer": {
            "steps": [
                {"title": "Production Function", "content": "Q = L^{\\\\alpha} K^{1-\\\\alpha}"},
                {"title": "Cost Function", "content": "C = wL + rK"},
                {"title": "Cost Minimization", "content": "\\\\min_{L,K} wL + rK \\\\quad \\\\text{s.t.} \\\\quad L^{\\\\alpha} K^{1-\\\\alpha} = Q"},
                {"title": "Technical Rate of Substitution", "content": "TRS = \\\\frac{MP_L}{MP_K} = \\\\frac{\\\\alpha K}{(1-\\\\alpha) L} = \\\\frac{w}{r}"},
                {"title": "Conditional Labor Demand", "content": "L^* = Q \\\\left(\\\\frac{\\\\alpha r}{(1-\\\\alpha) w}\\\\right)^{1-\\\\alpha}"}
            ],
            "final": "L^* = Q \\\\left(\\\\frac{\\\\alpha r}{(1-\\\\alpha) w}\\\\right)^{1-\\\\alpha}, \\\\quad K^* = Q \\\\left(\\\\frac{(1-\\\\alpha) w}{\\\\alpha r}\\\\right)^{\\\\alpha}"
        },
        "market": {
            "steps": [
                {"title": "Demand Function", "content": "Q_d = a - bP"},
                {"title": "Supply Function", "content": "Q_s = c + dP"},
                {"title": "Equilibrium Condition", "content": "Q_d = Q_s"},
                {"title": "Equilibrium Equation", "content": "a - bP = c + dP"},
                {"title": "Equilibrium Price", "content": "P^* = \\\\frac{a - c}{b + d}"}
            ],
            "final": "P^* = \\\\frac{a - c}{b + d}, \\\\quad Q^* = \\\\frac{ad + bc}{b + d}"
        },
        "game": {
            "steps": [
                {"title": "Payoff Matrix", "content": "\\\\begin{pmatrix} (3,3) & (0,5) \\\\\\\\ (5,0) & (1,1) \\\\end{pmatrix}"},
                {"title": "Nash Equilibrium", "content": "Finding best responses"},
                {"title": "Strategy Analysis", "content": "Mutual defection is an equilibrium"},
                {"title": "Prisoner's Dilemma", "content": "Equilibrium is not socially optimal"}
            ],
            "final": "Nash Equilibrium: (Defect, Defect) with payoffs (1,1)"
        }
    }
    
    response_data = demo_responses.get(problem_type, {
        "steps": [{"title": "Demo Mode", "content": "Use your own API key for personalized AI solutions"}],
        "final": "Solution in demo mode - Add your API key for AI"
    })
    
    return ProblemResponse(
        steps=[Step(**step) for step in response_data["steps"]],
        final=response_data["final"],
        provider="demo"
    )

@app.get("/")
async def root():
    return {
        "message": "Microeconomics AI Solver API", 
        "version": "2.0.0",
        "status": "healthy",
        "note": "Users provide their own API keys"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "microeconomics-ai-solver"}

@app.post("/solve", response_model=ProblemResponse)
async def solve_problem(request: ProblemRequest):
    try:
        logger.info(f"Solving problem of type: {request.type} with provider: {request.provider}")
        
        # Vérifier si l'utilisateur a fourni une clé API
        if not request.user_api_key or not request.user_api_key.strip():
            logger.info("No API key provided, using demo mode")
            return get_demo_response(request.type, request.params)
        
        # Valider le format de la clé API
        if not request.user_api_key.startswith('sk-'):
            raise Exception("Invalid API key format. API keys usually start with 'sk-'.")
        
        # Construire le prompt
        prompt = build_prompt(request.type, request.params, request.description, request.settings)
        
        # Appeler l'API avec la clé utilisateur
        if request.provider == "deepseek":
            api_response = call_deepseek_api(prompt, request.user_api_key.strip())
        elif request.provider == "openai":
            api_response = call_openai_api(prompt, request.user_api_key.strip())
        else:
            return get_demo_response(request.type, request.params)
        
        # Traiter la réponse
        content = api_response["choices"][0]["message"]["content"]
        result = parse_api_response(content)
        result.provider = request.provider
        
        # Ajouter les informations d'usage des tokens
        if "usage" in api_response:
            result.token_usage = {
                "total": api_response["usage"]["total_tokens"],
                "prompt": api_response["usage"]["prompt_tokens"],
                "completion": api_response["usage"]["completion_tokens"]
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Error solving problem: {str(e)}")
        # Retourner une réponse d'erreur structurée
        return ProblemResponse(
            steps=[Step(title="Error", content=f"Error with {request.provider}: {str(e)}")],
            final=f"Cannot solve problem: {str(e)}",
            provider=request.provider,
            success=False
        )

def build_prompt(problem_type: str, params: dict, description: str = "", settings: dict = None) -> str:
    """Construit le prompt pour l'API"""
    
    prompt = f"Solve this microeconomics problem of type {problem_type}.\n\n"
    
    if problem_type == "consumer":
        prompt += f"Utility function: {params.get('utility', 'Not specified')}\n"
        prompt += f"Budget constraint: {params.get('constraint', 'Not specified')}\n"
    elif problem_type == "producer":
        prompt += f"Production function: {params.get('production', 'Not specified')}\n"
        prompt += f"Cost constraint: {params.get('cost', 'Not specified')}\n"
        prompt += f"Target output: {params.get('target_output', 'Not specified')}\n"
    elif problem_type == "market":
        prompt += f"Demand function: {params.get('demand', 'Not specified')}\n"
        prompt += f"Supply function: {params.get('supply', 'Not specified')}\n"
    elif problem_type == "game":
        prompt += f"Payoff matrix: {params.get('payoffs', 'Not specified')}\n"
    
    if description:
        prompt += f"\nAdditional description: {description}\n"
    
    prompt += """
    \nProvide the solution in the following JSON format:
    {
        "steps": [
            {"title": "Step title", "content": "LaTeX content"},
            ...
        ],
        "final": "Final answer in LaTeX"
    }
    
    Use $$ for display formulas and $ for inline formulas.
    """
    
    return prompt

def parse_api_response(response: str) -> ProblemResponse:
    """Parse la réponse de l'API"""
    try:
        data = json.loads(response)
        
        if "steps" not in data or "final" not in data:
            raise ValueError("Invalid response format")
        
        steps = []
        for step_data in data["steps"]:
            if isinstance(step_data, dict) and "title" in step_data and "content" in step_data:
                steps.append(Step(title=step_data["title"], content=step_data["content"]))
        
        return ProblemResponse(steps=steps, final=data["final"])
        
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse API response as JSON: {e}")
        # Fallback: créer une réponse basique
        return ProblemResponse(
            steps=[Step(title="AI Response", content=response)],
            final=response[:200] + "..." if len(response) > 200 else response
        )

# Pour le déploiement sur Render
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

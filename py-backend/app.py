from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum
from typing import List
from datetime import datetime, timezone
import uuid, io
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from torch import nn
from PIL import Image
import re

# ===== Code Analyzer =====
class CodeAnalyzer:
    def __init__(self):
        # Define comment patterns for different languages
        self.comment_patterns = {
            'python': [
                r'^\s*#.*$',  # Single line comments
                r'^\s*""".*?"""\s*$',  # Multi-line string comments (docstrings)
                r"^\s*'''.*?'''\s*$",  # Multi-line string comments (docstrings)
            ],
            'javascript': [
                r'^\s*//.*$',  # Single line comments
                r'^\s*/\*.*?\*/\s*$',  # Multi-line comments
            ],
            'java': [
                r'^\s*//.*$',  # Single line comments
                r'^\s*/\*.*?\*/\s*$',  # Multi-line comments
            ],
            'c_cpp': [
                r'^\s*//.*$',  # Single line comments
                r'^\s*/\*.*?\*/\s*$',  # Multi-line comments
            ],
            'csharp': [
                r'^\s*//.*$',  # Single line comments
                r'^\s*/\*.*?\*/\s*$',  # Multi-line comments
                r'^\s*///.*$',  # XML documentation comments
            ],
            'ruby': [
                r'^\s*#.*$',  # Single line comments
                r'^\s*=begin.*?=end\s*$',  # Multi-line comments
            ],
            'php': [
                r'^\s*//.*$',  # Single line comments
                r'^\s*#.*$',   # Shell-style comments
                r'^\s*/\*.*?\*/\s*$',  # Multi-line comments
            ],
            'html': [
                r'^\s*<!--.*?-->\s*$',  # HTML comments
            ],
            'css': [
                r'^\s*/\*.*?\*/\s*$',  # CSS comments
            ],
            'sql': [
                r'^\s*--.*$',  # Single line comments
                r'^\s*/\*.*?\*/\s*$',  # Multi-line comments
            ]
        }
        
        # Define function patterns for different languages
        self.function_patterns = {
            'python': [
                r'^\s*def\s+\w+\s*\([^)]*\)\s*:',  # Function definitions
                r'^\s*class\s+\w+\s*\([^)]*\)\s*:',  # Class definitions
                r'^\s*lambda\s+[^:]+:',  # Lambda functions
            ],
            'javascript': [
                r'^\s*function\s+\w*\s*\([^)]*\)\s*{',  # Function declarations
                r'^\s*\w+\s*=\s*function\s*\([^)]*\)\s*{',  # Function expressions
                r'^\s*\w+\s*=\s*\([^)]*\)\s*=>\s*{',  # Arrow functions
                r'^\s*class\s+\w+\s*{',  # Class declarations
            ],
            'java': [
                r'^\s*(public|private|protected)\s+\w+\s+\w+\s*\([^)]*\)\s*{',  # Method definitions
                r'^\s*class\s+\w+\s*{',  # Class definitions
            ],
            'c_cpp': [
                r'^\s*\w+\s+\w+\s*\([^)]*\)\s*{',  # Function definitions
                r'^\s*class\s+\w+\s*{',  # Class definitions
            ],
            'csharp': [
                r'^\s*(public|private|protected)\s+\w+\s+\w+\s*\([^)]*\)\s*{',  # Method definitions
                r'^\s*class\s+\w+\s*{',  # Class definitions
            ]
        }
        
        # General patterns that work across multiple languages
        self.general_comment_patterns = [
            r'^\s*#.*$',      # Python, Ruby, Perl, etc.
            r'^\s*//.*$',     # JavaScript, Java, C++, C#, etc.
            r'^\s*///.*$',    # C# XML comments
            r'^\s*/\*.*?\*/\s*$',  # Multi-line comments
            r'^\s*--.*$',     # SQL, Haskell, etc.
            r'^\s*<!--.*?-->\s*$',  # HTML comments
        ]
        
        self.general_function_patterns = [
            r'^\s*def\s+\w+',        # Python functions
            r'^\s*function\s+\w*',   # JavaScript functions
            r'^\s*\w+\s+[\w<>]+\s*\([^)]*\)\s*{',  # C++/Java functions
            r'^\s*class\s+\w+',      # Class definitions
            r'^\s*\w+\s*=\s*\([^)]*\)\s*=>',  # Arrow functions
        ]

    def detect_language(self, code):
        """Detect programming language based on code patterns"""
        code_lower = code.lower()
        
        language_indicators = {
            'python': ['def ', 'import ', 'from ', 'print(', 'elif ', 'lambda '],
            'javascript': ['function ', 'var ', 'let ', 'const ', '=>', 'console.log'],
            'java': ['public class', 'private ', 'protected ', 'import java.'],
            'c_cpp': ['#include', 'using namespace', 'cout <<', 'printf('],
            'csharp': ['using System', 'namespace ', 'public partial'],
            'html': ['<!DOCTYPE', '<html', '<div ', '<p ', '</'],
            'css': ['body {', '.class', '#id', 'font-size:'],
            'sql': ['SELECT ', 'FROM ', 'WHERE ', 'INSERT INTO'],
            'php': ['<?php', '$', 'echo ', 'mysql_'],
            'ruby': ['def ', 'end', 'puts ', 'require ']
        }
        
        for lang, indicators in language_indicators.items():
            for indicator in indicators:
                if indicator in code_lower:
                    return lang
        return 'unknown'

    def is_comment(self, line, language=None):
        """Check if a line is a comment"""
        line = line.strip()
        if not line:
            return False
            
        # Try language-specific patterns first
        if language and language in self.comment_patterns:
            for pattern in self.comment_patterns[language]:
                if re.match(pattern, line, re.DOTALL):
                    return True
        
        # Fall back to general patterns
        for pattern in self.general_comment_patterns:
            if re.match(pattern, line, re.DOTALL):
                return True
                
        return False

    def is_function(self, line, language=None):
        """Check if a line contains a function definition"""
        line = line.strip()
        if not line:
            return False
            
        # Try language-specific patterns first
        if language and language in self.function_patterns:
            for pattern in self.function_patterns[language]:
                if re.search(pattern, line):
                    return True
        
        # Fall back to general patterns
        for pattern in self.general_function_patterns:
            if re.search(pattern, line):
                return True
                
        return False

    def is_blank_line(self, line):
        """Check if a line is blank (only whitespace)"""
        return len(line.strip()) == 0

    def analyze_code(self, code):
        """Main function to analyze code and extract metrics"""
        lines = code.split('\n')
        language = self.detect_language(code)
        
        metrics = {
            'lines': len(lines),
            'code_lines': 0,
            'comment_lines': 0,
            'blank_lines': 0,
            'functions': 0,
            'language': language
        }
        
        in_multiline_comment = False
        multiline_comment_delimiter = None
        
        for line in lines:
            # Check for blank lines
            if self.is_blank_line(line):
                metrics['blank_lines'] += 1
                continue
                
            # Handle multi-line comments
            if in_multiline_comment:
                metrics['comment_lines'] += 1
                # Check if this line ends the multi-line comment
                if multiline_comment_delimiter == '"""' and '"""' in line:
                    in_multiline_comment = False
                elif multiline_comment_delimiter == "'''" and "'''" in line:
                    in_multiline_comment = False
                elif multiline_comment_delimiter == '*/' and '*/' in line:
                    in_multiline_comment = False
                elif multiline_comment_delimiter == '=end' and '=end' in line:
                    in_multiline_comment = False
                elif multiline_comment_delimiter == '-->' and '-->' in line:
                    in_multiline_comment = False
                continue
            
            # Check for start of multi-line comments
            if language == 'python':
                if line.strip().startswith('"""') and line.strip().endswith('"""') and len(line.strip()) > 3:
                    metrics['comment_lines'] += 1
                    continue
                elif line.strip().startswith('"""'):
                    in_multiline_comment = True
                    multiline_comment_delimiter = '"""'
                    metrics['comment_lines'] += 1
                    continue
                elif line.strip().startswith("'''") and line.strip().endswith("'''") and len(line.strip()) > 3:
                    metrics['comment_lines'] += 1
                    continue
                elif line.strip().startswith("'''"):
                    in_multiline_comment = True
                    multiline_comment_delimiter = "'''"
                    metrics['comment_lines'] += 1
                    continue
            elif language in ['javascript', 'java', 'c_cpp', 'csharp', 'css', 'sql']:
                if '/*' in line and '*/' in line:
                    metrics['comment_lines'] += 1
                    continue
                elif '/*' in line:
                    in_multiline_comment = True
                    multiline_comment_delimiter = '*/'
                    metrics['comment_lines'] += 1
                    continue
            elif language == 'ruby':
                if line.strip().startswith('=begin'):
                    in_multiline_comment = True
                    multiline_comment_delimiter = '=end'
                    metrics['comment_lines'] += 1
                    continue
            elif language == 'html':
                if '<!--' in line and '-->' in line:
                    metrics['comment_lines'] += 1
                    continue
                elif '<!--' in line:
                    in_multiline_comment = True
                    multiline_comment_delimiter = '-->'
                    metrics['comment_lines'] += 1
                    continue
            
            # Check for single-line comments
            if self.is_comment(line, language):
                metrics['comment_lines'] += 1
            # Check for functions
            elif self.is_function(line, language):
                metrics['functions'] += 1
                metrics['code_lines'] += 1
            else:
                metrics['code_lines'] += 1
        
        # Calculate ratios for the logistic regression model
        if metrics['lines'] > 0:
            metrics['F.L'] = metrics['functions'] / metrics['lines']  # Functions per line
            metrics['CM.L'] = metrics['comment_lines'] / metrics['lines']  # Comments per line
            metrics['B.L'] = metrics['blank_lines'] / metrics['lines']  # Blank lines per line
        else:
            metrics['F.L'] = 0
            metrics['CM.L'] = 0
            metrics['B.L'] = 0
            
        if metrics['code_lines'] > 0:
            metrics['CM.CD'] = metrics['comment_lines'] / metrics['code_lines']  # Comments per code line
        else:
            metrics['CM.CD'] = 0
            
        metrics['FN'] = metrics['functions']  # Number of functions
        metrics['B'] = metrics['blank_lines']  # Number of blank lines
        
        return metrics

    def predict_ai_probability(self, metrics):
        """Use your logistic regression model to predict AI probability"""
        # Your logistic regression coefficients
        F_L = metrics['F.L']
        FN = metrics['FN']
        CM_CD = metrics['CM.CD']
        CM_L = metrics['CM.L']
        B = metrics['B']
        
        # Logistic regression equation with your coefficients
        z = (-0.54119435 + 
             11.39265578 * F_L + 
             -0.20647395 * FN + 
             21.43916272 * CM_CD + 
             -24.27715020 * CM_L + 
             -0.05280026 * B)
 
        
        raw_probability = 1 / (1 + 2.71828 ** (-z))
        probability = raw_probability

        # This makes the threshold the midpoint instead of 0.5
        if (raw_probability < 2 * 0.397477):
            normalized_probability = raw_probability / (2 * 0.397477)
            probability = normalized_probability      
   
        return probability

# ===== Enums =====
class Verdict(str, Enum):
    LIKELY_HUMAN = "LIKELY_HUMAN"
    LIKELY_AI = "LIKELY_AI"

class SourceType(str, Enum):
    PLAINTEXT = "PLAINTEXT"

# ===== Entities =====
class Submission(BaseModel):
    id: str
    sourceType: SourceType
    rawText: str
    uploadedAt: str
    tokenCount: int

    @staticmethod
    def calculateTokenCount(text: str) -> int:
        return len((text or "").split())

class Highlight(BaseModel):
    startIndex: int
    endIndex: int
    score: float

class AnalysisResult(BaseModel):
    id: str
    verdict: Verdict
    modelVersion: str
    createdAt: str
    highlights: List[Highlight] = []
    score: float = 0.0

    def getSummary(self) -> str:
        return f"{self.verdict} via {self.modelVersion}"

    def getHighlights(self) -> List[Highlight]:
        return self.highlights

# ===== HuggingFace model class =====
class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        # Initialize the base transformer model.
        self.model = AutoModel.from_config(config)
        # Define a classifier head.
        self.classifier = nn.Linear(config.hidden_size, 1)
        # Initialize weights (handled by PreTrainedModel)
        self.init_weights()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Forward pass through the transformer
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        # Classifier
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())

        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output

# ===== Services =====
class AnalysisService:
    def __init__(self,
                 model_directory="desklib/ai-text-detector-v1.01",
                 threshold=0.5,
                 max_len=1800):
        self.model_directory = model_directory
        self.threshold = threshold
        self.max_len = max_len
        self.code_analyzer = CodeAnalyzer()

        print(f"Loading model {model_directory} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_directory)
        self.model = DesklibAIDetectionModel.from_pretrained(model_directory)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def run_llm_analysis(self, text: str) -> float:
        """Run the transformer model analysis and return probability"""
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            probability = torch.sigmoid(logits).item()
        
        return probability

    def run_statistical_analysis(self, text: str) -> float:
        """Run the statistical code analysis and return probability"""
        metrics = self.code_analyzer.analyze_code(text)
        probability = self.code_analyzer.predict_ai_probability(metrics)
        return probability

    def runAnalysis(self, submission: Submission, is_code_mode: bool = False) -> AnalysisResult:
        text = submission.rawText

        # If in code mode, combine both models with 60/40 weighting
        if is_code_mode:
            print("Using combined code analysis mode (60% LLM, 40% Statistical)...")
            
            # Get probabilities from both models
            llm_probability = self.run_llm_analysis(text)
            statistical_probability = self.run_statistical_analysis(text)
            
           
            if(llm_probability >= 0.6 and statistical_probability < 0.5):
                statistical_probability = 1 - statistical_probability
            
            # Combine with 60/40 weighting
            combined_probability = (0.6 * llm_probability) + (0.4 * statistical_probability)
            
            print(f"LLM probability: {llm_probability:.4f}")
            print(f"Statistical probability: {statistical_probability:.4f}")
            print(f"Combined probability (40/60): {combined_probability:.4f}")
            
            # Use threshold of 0.5 for final decision
            predicted_label = 1 if combined_probability >= self.threshold else 0
            verdict = Verdict.LIKELY_AI if predicted_label == 1 else Verdict.LIKELY_HUMAN
            
            return AnalysisResult(
                id=str(uuid.uuid4()),
                verdict=verdict,
                modelVersion="combined-llm-statistical-model",
                createdAt=datetime.now(timezone.utc).isoformat(),
                highlights=[],
                score=round(combined_probability * 100)
            )
        
        # Otherwise, use only the transformer model (original behavior)
        llm_probability = self.run_llm_analysis(text)
        print(f"Text analysis - AI probability: {llm_probability*100:.2f}%")
        
        predicted_label = 1 if llm_probability >= self.threshold else 0
        verdict = Verdict.LIKELY_AI if predicted_label == 1 else Verdict.LIKELY_HUMAN
        print(f"Verdict: {verdict}")
        
        highlights: List[Highlight] = []

        return AnalysisResult(
            id=str(uuid.uuid4()),
            verdict=verdict,
            modelVersion=self.model_directory,
            createdAt=datetime.now(timezone.utc).isoformat(),
            highlights=highlights,
            score=round(llm_probability*100)
        )
    
        
        
class SubmissionService:
    MAX_TOKENS: int = 5000

    def __init__(self, analysis_service: AnalysisService):
        self.analysis_service = analysis_service

    def ingestText(self, text: str) -> Submission:
        text = (text or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text is required.")
        if not self.validateLength(text):
            raise HTTPException(status_code=400, detail=f"Maximum words: {self.MAX_TOKENS}.")
        submission = Submission(
            id=str(uuid.uuid4()),
            sourceType=SourceType.PLAINTEXT,
            rawText=text,
            uploadedAt=datetime.now(timezone.utc).isoformat(),
            tokenCount=Submission.calculateTokenCount(text),
        )
        self.emitForAnalysis(submission.id)
        return submission

    def validateLength(self, text: str) -> bool:
        return Submission.calculateTokenCount(text) <= self.MAX_TOKENS

    def emitForAnalysis(self, submissionId: str) -> None:
        return None
    

# ===== Image Analysis Service =====
class ImageAnalysisService:
    
    def __init__(self, model_directory: str = "haywoodsloan/ai-image-detector-deploy"):
        self.model_directory = model_directory
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None

        try:
            self.processor = AutoImageProcessor.from_pretrained(model_directory, use_fast=False)
            self.model = AutoModelForImageClassification.from_pretrained(model_directory)
            self.model.to(self.device)
            self.model.eval()
            print(f"Image model loaded from {model_directory} on {self.device}")
        except Exception as e:
            print(f"Warning: failed to load image model from {model_directory}: {e}")
            self.processor = None
            self.model = None
 
    def predict_probability(self, pil_image) -> int:
        inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits        
        
        prediction = logits.argmax(-1).item()  
        print(self.model.config.id2label[prediction])
        return prediction

# ===== FastAPI wiring =====
app = FastAPI(title="Scout AI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

analysis_service = AnalysisService()  # loads the HuggingFace model
submission_service = SubmissionService(analysis_service)
image_service = ImageAnalysisService()

class TextPayload(BaseModel):
    text: str
    isCodeMode: bool = False

@app.get("/api/health")
def health():
    return "OK"

@app.post("/api/submit-text", response_model=AnalysisResult)
def submit_text(payload: TextPayload):
    submission = submission_service.ingestText(payload.text)
    result = analysis_service.runAnalysis(submission, payload.isCodeMode)
    
    # Log the detection mode for debugging
    print(f"Detection mode: {'Combined Code Analysis' if payload.isCodeMode else 'Text Only'}")
    
    return result

@app.post("/api/submit-image", response_model=AnalysisResult)
async def submit_image(image: UploadFile = File(...)):
    if image is None or image.filename is None:
        raise HTTPException(status_code=400, detail="No image uploaded")
    try:
        content = await image.read()
        pil = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read image: {e}")

    pred = image_service.predict_probability(pil)
    verdict = 0
    
    if pred == 0:
        verdict = Verdict.LIKELY_AI
    else:
        verdict = Verdict.LIKELY_HUMAN
    
    print(verdict)

    result = AnalysisResult(
        id=str(uuid.uuid4()),
        verdict=verdict,
        modelVersion=image_service.model_directory,
        createdAt=datetime.now(timezone.utc).isoformat(),
    )
    return result

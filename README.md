# DSPy: Build and Optimize Agentic Apps

> Learning materials from DeepLearning.AI's DSPy course taught by Chen Qian (Databricks)

[![Course](https://img.shields.io/badge/Course-DeepLearning.AI-blue)](https://www.deeplearning.ai/short-courses/dspy-build-and-optimize-agentic-apps/)
[![DSPy](https://img.shields.io/badge/Framework-DSPy-orange)](https://dspy.ai/)
[![MLflow](https://img.shields.io/badge/Tracing-MLflow-green)](https://mlflow.org/)

## 📚 Course Overview

A 49-minute intermediate course covering DSPy's signature-based programming model for building modular, traceable, and optimizable GenAI agentic applications.

### What is DSPy?

DSPy is an open-source framework that simplifies LLM application development by:
- Replacing manual prompt engineering with **programmatic signatures**
- Enabling **automatic optimization** of prompts and few-shot examples
- Providing **model-agnostic** interfaces (switch between OpenAI, Gemini, Groq, etc.)

---

## 🎯 Learning Objectives

| Lesson | Topic | Key Concepts |
|--------|-------|--------------|
| **L1** | Introduction to DSPy | Signature vs. traditional prompting |
| **L2** | Signatures & Modules | `Predict`, `ChainOfThought`, Custom Modules |
| **L3** | MLflow Tracing | Debugging ReAct agents, hierarchical traces |
| **L4** | DSPy Optimizer | Automated prompt tuning, RAG optimization |

---

## 🛠️ Projects Implemented

### 1. Sentiment Analyzer (Lesson 2)
```python
# String-based signature
classifier = dspy.Predict("text -> sentiment, confidence: float")
result = classifier(text="I love this product!")
```

**Models Used**: Gemini Gemma 3-12B-IT, Gemini 2.5 Flash

### 2. Airline Customer Service Agent (Lesson 3)
```python
# ReAct pattern with tools
agent = dspy.ReAct(
    signature=AirlineAgentSignature,
    tools=[search_flights, book_flight, create_support_ticket],
    max_iters=10
)
```

**Features**: Multi-hop reasoning, tool calling, MLflow tracing

### 3. Wikipedia RAG Agent (Lesson 4)
- **Baseline accuracy**: 31% exact match
- **After optimization**: 54% exact match
- **Method**: DSPy Optimizer with labeled examples

---

## 🚀 Quick Start

### Installation
```bash
pip install dspy-ai mlflow python-dotenv
```

### Basic Usage
```python
import dspy
import mlflow

# Configure LLM
lm = dspy.LM('gemini/gemini-2.5-flash', api_key="YOUR_KEY")
dspy.configure(lm=lm)

# Enable tracing
mlflow.dspy.autolog()

# Define signature
class TaskSignature(dspy.Signature):
    """Your task description"""
    input_field: str = dspy.InputField()
    output_field: str = dspy.OutputField()

# Build module
predictor = dspy.ChainOfThought(TaskSignature)
result = predictor(input_field="Your query")
```

---

## 📊 Key Learnings

### Module Comparison

| Module | Use Case | Reasoning | Cost |
|--------|----------|-----------|------|
| `Predict` | Simple Q&A | ❌ None | $ |
| `ChainOfThought` | Complex tasks | ✅ Yes | $$ |
| `ReAct` | Multi-step agents | ✅ + Tools | $$$ |

### Model Comparison (from experiments)

| Provider | Model | Speed | Cost | Best For |
|----------|-------|-------|------|----------|
| Gemini | Gemma 3-12B-IT | Moderate | Low | Prototyping |
| Gemini | Gemini 2.5 Flash | Fast | Low | Production |
| Groq | Llama 3-70B | **Very Fast** | Low | Real-time apps |

---

## 💡 Real-World Applications

1. **Customer Support** - Automate 80% of booking/cancellation requests
2. **Financial Analysis** - Portfolio recommendations with audit trails
3. **E-commerce** - Conversational product search with 40% better conversion
4. **Healthcare** - Prior authorization with HIPAA-compliant tracing

---

## 🔍 MLflow Tracing Benefits

```python
# One line enables full observability
mlflow.dspy.autolog()

# Captures 4 layers:
# 1. Module calls (ReAct, Predict)
# 2. Adapter formatting (prompt construction)
# 3. LLM interactions (raw requests/responses)
# 4. Tool executions (function calls)
```

**Production Value**:
- 🐛 Debug multi-hop reasoning failures
- 📈 Monitor success rates and latency
- 💰 Optimize token usage (saved 40% in experiments)
- 📋 Compliance audit trails

---

## 📁 Repository Structure

```
.
├── lesson_2_signatures_modules/
│   ├── sentiment_classifier.ipynb
│   └── custom_module_example.py
├── lesson_3_mlflow_tracing/
│   ├── airline_agent.ipynb
│   └── trace_analysis.md
├── lesson_4_optimizer/
│   └── rag_optimization.ipynb
└── README.md
```

---

## 🎓 Course Details

- **Duration**: 49 minutes
- **Level**: Intermediate
- **Format**: 6 video lessons + 3 hands-on labs
- **Instructor**: Chen Qian (Databricks, DSPy co-maintainer)
- **Prerequisites**: Basic Python, familiarity with LLMs

---

## 🔗 Resources

- [DSPy Documentation](https://dspy.ai/)
- [MLflow Docs](https://mlflow.org/docs/latest/genai/tracing/)
- [Course Link](https://www.deeplearning.ai/short-courses/dspy-build-and-optimize-agentic-apps/)
- [Databricks MLflow](https://www.databricks.com/product/managed-mlflow)

---

## 📝 Key Takeaways

1. **Signatures replace prompts** - Define input/output contracts, let DSPy handle formatting
2. **Modules are composable** - Build complex agents from simple building blocks
3. **Tracing is essential** - MLflow makes multi-step agents debuggable
4. **Optimization is automated** - DSPy Optimizer improves accuracy without manual tuning
5. **Model-agnostic design** - Switch providers with one line of code

---

## 🙏 Acknowledgments

- **DeepLearning.AI** for course platform
- **Databricks** for partnership and MLflow integration
- **DSPy Community** for open-source framework

---

## 📄 License

Course materials are for educational purposes. Refer to [DeepLearning.AI Terms](https://www.deeplearning.ai/terms/) for usage policies.

---

**⭐ If this helped your learning, consider starring this repository!**
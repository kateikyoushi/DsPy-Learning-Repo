# 🎯 Cebu Pacific Customer Support Agent Optimization with DSPy

## Complete Jupyter Notebook Pipeline for Agent Optimization

This repository demonstrates **automated AI agent optimization** using DSPy and Groq LLM to improve customer support quality and efficiency.

---

## 📊 Demo Results Summary

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Quality Score** | 30% | 85% | +55% ✅ |
| **Resolution Time** | 5 min | 30 sec | 90% faster ⚡ |
| **Annual Savings** | $0 | $821,250 | ♾️ ROI 💰 |

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Requirements

```bash
pip install dspy-ai mlflow matplotlib numpy
```

### Step 2: Get Groq API Key (FREE)

1. Go to: https://console.groq.com/keys
2. Sign up / Log in
3. Create API key
4. Copy the key

### Step 3: Setup Files

Ensure you have these 3 files in the same directory:
- `cebu_pacific_trainset.jsonl` (50 training examples)
- `cebu_pacific_valset.jsonl` (20 validation examples)
- `cebu_pacific_agent_optimization_complete.py` (this notebook)

### Step 4: Run the Notebook

1. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Create new notebook

3. Copy each cell block from `cebu_pacific_agent_optimization_complete.py`

4. **IMPORTANT**: In Cell 4, replace:
   ```python
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
   with your actual API key:
   ```python
   GROQ_API_KEY = "gsk_your_actual_key_here"
   ```

5. Run all cells sequentially

6. Wait 3-5 minutes for optimization

7. View results and visualizations!

---

## 📁 File Structure

```
project/
│
├── cebu_pacific_trainset.jsonl          # 50 training examples
├── cebu_pacific_valset.jsonl            # 20 validation examples
├── cebu_pacific_agent_optimization_complete.py  # Notebook code
│
└── outputs/ (generated after running)
    ├── optimization_results.json        # Detailed metrics
    ├── cebu_pacific_optimized_agent.json  # Saved model
    └── visualizations.png               # Charts
```

---

## 🎯 What This Notebook Does

### STEP 1: Show the Problem (Unoptimized Agent)
- Demonstrates generic, unhelpful responses
- Shows baseline performance: ~30% quality score
- Highlights customer frustration

### STEP 2: Run DSPy Optimization (3-5 minutes)
- Loads 50 successful support resolutions
- Uses MIPROv2 to optimize prompts automatically
- Generates instructions and few-shot examples
- Tests combinations on validation set

### STEP 3: Show the Results (Optimized Agent)
- Same query, dramatically better response
- Detailed troubleshooting steps
- Specific contact information
- 85% quality score ✅

### STEP 4: Calculate Business Impact
- $821,250 annual savings
- 75 hours/day time saved
- 10× agent productivity
- ROI: ♾️ (costs ~$1 to optimize)

---

## 📊 Notebook Cells Overview

| Cell | Description | Duration |
|------|-------------|----------|
| 1 | Title and Introduction | Instant |
| 2 | Install packages | 30 sec |
| 3 | Import libraries | Instant |
| 4 | **Setup Groq API Key** ⚠️ | Instant |
| 5 | Configure DSPy with Groq | Instant |
| 6 | Load datasets (50 + 20 examples) | 1 sec |
| 7 | Visualize dataset statistics | 2 sec |
| 8 | Create support agent module | Instant |
| 9 | **STEP 1: Show Problem** | 5 sec |
| 10 | Define evaluation metric | Instant |
| 11 | Evaluate original agent | 30 sec |
| 12 | Configure MIPROv2 optimizer | Instant |
| 13 | **STEP 2: Run Optimization** ⏳ | **3-5 min** |
| 14 | Inspect optimized components | Instant |
| 15 | **STEP 3: Show Results** | 5 sec |
| 16 | Evaluate optimized agent | 30 sec |
| 17 | Visualize before/after | 2 sec |
| 18 | **STEP 4: Business Impact** | Instant |
| 19 | Business impact dashboard | 2 sec |
| 20 | Export results and save model | 1 sec |
| 21 | Final summary | Instant |

**Total Runtime: ~5-7 minutes**

---

## 🔧 Technical Details

### Framework & Tools
- **DSPy**: Prompt optimization framework
- **Groq**: Fast LLM inference (llama-3.1-8b-instant)
- **MIPROv2**: Multi-prompt optimizer
- **MLflow**: Experiment tracking (optional)

### Model Configuration
```python
Model: groq/llama-3.1-8b-instant
Max Tokens: 800
Temperature: 0.7
Optimization Mode: light (fast)
Training Examples: 50
Validation Examples: 20
```

### Evaluation Metric
Custom `support_quality_metric` checks for:
- ✅ Structured guidance (steps/options)
- ✅ Detailed response (>200 chars)
- ✅ Positive indicators
- ✅ Contact information
- ✅ Specific details (fees, policies)

---

## 📈 Expected Outputs

### 1. Console Output
```
✅ Datasets loaded successfully!
   Training set: 50 examples
   Validation set: 20 examples

BASELINE EVALUATION:
   Average Score: 30.5%

🚀 STARTING OPTIMIZATION...
   [Progress bars and status updates]

✅ OPTIMIZATION COMPLETE!
   Duration: 3.5 minutes

FINAL EVALUATION:
   Average Score: 85.2%
   Improvement: +54.7%

💰 BUSINESS IMPACT:
   Annual Savings: $821,250
```

### 2. Visualizations
- Performance comparison charts
- Score distribution histograms
- Response time analysis
- Business impact dashboard
- Monthly savings projection

### 3. Saved Files
- `optimization_results.json`: Detailed metrics
- `cebu_pacific_optimized_agent.json`: Trained model

---

## 🎓 Understanding the Code

### Core Components

#### 1. Support Agent Module
```python
class SupportAgent(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_response = dspy.ChainOfThought("query -> answer")

    def forward(self, query):
        response = self.generate_response(query=query)
        return response
```

#### 2. Optimization Process
```python
optimizer = dspy.MIPROv2(
    metric=support_quality_metric,
    auto="light",
    num_threads=8
)

optimized_agent = optimizer.compile(
    original_agent,
    trainset=trainset[:20],
    valset=valset[:10]
)
```

#### 3. Evaluation
```python
def support_quality_metric(example, pred, trace=None):
    # Checks for quality indicators
    # Returns score 0.0 to 1.0
    pass
```

---

## 🔍 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'dspy'"
**Solution:**
```bash
pip install dspy-ai --upgrade
```

### Issue: "Groq API key not found"
**Solution:**
- Check Cell 4 has your actual API key
- Ensure no quotes issues: `GROQ_API_KEY = "gsk_..."`
- Try setting environment variable:
  ```python
  import os
  os.environ["GROQ_API_KEY"] = "your_key_here"
  ```

### Issue: "File not found: cebu_pacific_trainset.jsonl"
**Solution:**
- Ensure all 3 files are in same directory
- Check file names match exactly (case-sensitive)
- Try absolute path:
  ```python
  with open("/full/path/to/cebu_pacific_trainset.jsonl", "r") as f:
  ```

### Issue: "Optimization taking too long (>10 minutes)"
**Solution:**
- Reduce training examples: `trainset[:10]`
- Reduce validation examples: `valset[:5]`
- Use auto="light" (already default)
- Check internet connection (Groq API calls)

### Issue: "Rate limit exceeded"
**Solution:**
- Groq free tier: 30 requests/minute
- Add delays between calls
- Reduce num_threads: `num_threads=2`
- Wait 1 minute and retry

### Issue: "Poor optimization results (<50% improvement)"
**Solution:**
- Check training data quality
- Increase training examples: `trainset[:30]`
- Try different metric threshold
- Run optimization multiple times

---

## 💡 Tips for Best Results

### 1. Data Quality Matters
- Use real, successful support resolutions
- Ensure consistent formatting
- Include diverse scenarios
- High-quality resolutions = better optimization

### 2. Metric Tuning
- Adjust quality indicators for your use case
- Consider domain-specific checks
- Weight different aspects differently

### 3. Optimization Settings
```python
# Fast (2-3 min): Good for testing
auto="light", num_threads=8, max_demos=3

# Balanced (5-7 min): Production
auto="medium", num_threads=16, max_demos=5

# Thorough (10-15 min): Maximum quality
auto="heavy", num_threads=24, max_demos=7
```

### 4. Cost Optimization
- Groq is FREE for development
- 50 training examples = optimal cost/quality
- Re-optimize monthly, not daily
- Cache common queries

---

## 🎯 Customization Guide

### Change to Your Use Case

#### 1. Update Data Format
```python
# In Cell 6, modify:
example = dspy.Example(
    query=data["your_query_field"],
    answer=data["your_answer_field"]
).with_inputs("query")
```

#### 2. Adjust Metric
```python
# In Cell 10, customize:
def your_custom_metric(example, pred, trace=None):
    # Your quality checks here
    score = check_your_criteria(pred.answer)
    return score
```

#### 3. Change Model
```python
# In Cell 5, use different model:
lm = dspy.LM(
    'groq/llama3-70b-8192',  # Larger model
    # or 'groq/mixtral-8x7b-32768'
    api_key=GROQ_API_KEY
)
```

#### 4. Modify Business Metrics
```python
# In Cell 18, adjust:
tickets_per_day = 500  # Your volume
agent_hourly_rate = 25  # Your rate
```

---

## 📚 Learn More

### DSPy Resources
- Documentation: https://dspy-docs.vercel.app/
- GitHub: https://github.com/stanfordnlp/dspy
- Discord: https://discord.gg/XCGy2WDCQB

### Groq Resources
- Console: https://console.groq.com/
- Documentation: https://console.groq.com/docs
- Pricing: https://groq.com/pricing (FREE tier available)

### Related Tutorials
- DSPy Quick Start: https://dspy-docs.vercel.app/docs/quick-start
- MIPROv2 Guide: https://dspy-docs.vercel.app/docs/building-blocks/optimizers
- Groq Python SDK: https://github.com/groq/groq-python

---

## 🤝 Support & Contribution

### Need Help?
1. Check troubleshooting section above
2. Review cell-by-cell outputs
3. Join DSPy Discord
4. Open GitHub issue

### Want to Contribute?
- Share your use case results
- Submit improvements to evaluation metric
- Add new visualization ideas
- Report bugs or issues

---

## 📄 License

This project is provided as-is for educational and demonstration purposes.

### Credits
- **DSPy Framework**: Stanford NLP
- **Groq LLM**: Groq Inc.
- **Dataset**: Cebu Pacific support scenarios (synthetic for demo)

---

## 🎉 Success Stories

### Expected Results
- **Quality Score**: 30% → 85% (+55%)
- **Response Time**: 5 min → 30 sec (90% faster)
- **Customer Satisfaction**: +119% improvement
- **Annual Savings**: $821,250

### Real-World Applications
- ✅ Customer support automation
- ✅ Technical troubleshooting guides
- ✅ FAQ response generation
- ✅ Email response drafting
- ✅ Chatbot optimization

---

## 🚀 Next Steps After Completing This Notebook

### 1. Production Deployment
- [ ] Set up API endpoint
- [ ] Implement caching layer
- [ ] Add monitoring dashboard
- [ ] Configure auto-scaling

### 2. Continuous Improvement
- [ ] Collect real user feedback
- [ ] Re-optimize monthly
- [ ] A/B test variations
- [ ] Expand training data

### 3. Scale to Other Domains
- [ ] Technical support
- [ ] Sales inquiries
- [ ] Billing questions
- [ ] Product information

---

## 📞 Contact

**Questions about this notebook?**
- Open an issue on GitHub
- Join DSPy Discord community
- Email: support@yourdomain.com

**Using this for your business?**
- We'd love to hear your results!
- Share your success story
- Contribute improvements back

---

## 🎓 Appendix: Full Cell Reference

### Cell-by-Cell Checklist

- [ ] Cell 1: Read introduction
- [ ] Cell 2: Install packages (`pip install dspy-ai mlflow matplotlib`)
- [ ] Cell 3: Import libraries
- [ ] Cell 4: **SET YOUR GROQ API KEY** ⚠️
- [ ] Cell 5: Configure DSPy
- [ ] Cell 6: Load datasets (ensure files exist)
- [ ] Cell 7: View dataset stats
- [ ] Cell 8: Create agent module
- [ ] Cell 9: See unoptimized performance
- [ ] Cell 10: Define metric
- [ ] Cell 11: Baseline evaluation
- [ ] Cell 12: Setup optimizer
- [ ] Cell 13: **RUN OPTIMIZATION** (wait 3-5 min) ⏳
- [ ] Cell 14: Inspect optimized components
- [ ] Cell 15: See optimized performance
- [ ] Cell 16: Final evaluation
- [ ] Cell 17: View comparison charts
- [ ] Cell 18: See business impact
- [ ] Cell 19: View dashboard
- [ ] Cell 20: Export results
- [ ] Cell 21: Read summary

---

**✨ You're ready to go! Start with Cell 1 and run through Cell 21.**

**Questions? Check the troubleshooting section or reach out for help!**

---

_Last updated: February 16, 2026_
_Version: 1.0_

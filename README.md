# 🎮 AiCade-POC Backend

Welcome to the **AiCade-POC Backend** – your AI-powered code improvement engine, tailored for game developers! This backend leverages FastAPI and the efficient `deepseek-ai/deepseek-coder-1.3b-instruct` model, running locally with 4-bit quantization for lightning-fast, resource-friendly code suggestions.

---

## 🤖 Model & Approach

- **Model:** [`deepseek-ai/deepseek-coder-1.3b-instruct`](https://huggingface.co/deepseek-ai/deepseek-coder-1.3b-instruct)
- **Why this model?**
  - **Optimized for Game Dev:** Delivers actionable, performance-focused code improvements.
  - **Runs Locally:** No API keys, no cloud, no data leaves your machine.
  - **4-bit Quantization:** Uses `bitsandbytes` for ultra-low memory usage—perfect for 2GB GPUs and above.
  - **Fast & Private:** All processing happens on your hardware, ensuring privacy and speed.
  - **Right-Sized for Most PCs:** While a larger 6.7B parameter model exists and can provide even better results, it requires significantly more GPU memory (typically 8GB+). The 1.3B model is chosen here because it fits comfortably on systems with only 2GB of GPU VRAM, making it accessible to more developers.
  - **Low Latency & Resource Efficiency:** The 1.3B model loads quickly, responds fast, and consumes less power—ideal for rapid prototyping and frequent use.
  - **Sufficient Quality for Game Dev Tasks:** For most code improvement and suggestion tasks, especially in game development, the 1.3B model provides a strong balance of quality and efficiency.

> **Note:** If you have a more powerful GPU (8GB+ VRAM), you may experiment with the 6.7B model for potentially higher-quality suggestions, but for most users and typical hardware, the 1.3B model is the practical and reliable choice.

---

## 🚀 What Can It Do?

- **Smart Code Suggestions:** Input your code and a prompt—get back optimized code and a crisp technical explanation.
- **Language-Aware:** Detects and converts code to your target language if needed.
- **Game Performance Focus:** Suggestions are tuned for game development productivity and performance.

---

## ⚡ Quickstart (Windows)

1. **Create a Virtual Environment**
   ```sh
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install PyTorch**
   - **With GPU (CUDA):**
     ```sh
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
   - **CPU only:**
     ```sh
     pip install torch torchvision torchaudio
     ```

3. **Install Dependencies**
   ```sh
   pip install fastapi transformers uvicorn bitsandbytes
   ```

   > **Note:** `bitsandbytes` is essential for 4-bit quantized models.

---

## 🚦 Run the API

Start your backend with:

```sh
uvicorn app:app --reload
```

- API: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 💡 Tips

- **GPU Recommended:** For best results, use a CUDA-capable GPU (2GB+ VRAM).
- **Security:** Update `allow_origins` in `app.py` before deploying to production.
- **Troubleshooting:** Ensure your GPU drivers and CUDA toolkit are installed if using CUDA.

---

Supercharge your game development workflow with local, private, and blazing-fast AI code suggestions. Happy coding! 🚀

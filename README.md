---
title: AI Style Transfer
emoji: 🎨
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.31.5
app_file: app.py
pinned: false
---

# 🎨 Day 17: Neural Style Transfer with Gradio

This is the seventeenth project of my #30DaysOfAI challenge. This project explores **Generative AI** by implementing the classic Neural Style Transfer algorithm and deploying it with **Gradio** on Hugging Face Spaces.

### ✨ Key Concepts
* **Neural Style Transfer:** An optimization technique used to take two images—a content image and a style reference image—and blend them together so the output image retains the core content of the first image but is "painted" in the style of the second.
* **Pre-trained Models (VGG19):** The algorithm uses a pre-trained VGG19 network to extract content and style representations from the images.
* **Gradio for Deployment:** Instead of Streamlit, this project uses **Gradio**, a fast and simple framework for creating web UIs for machine learning models.

### 💻 Tech Stack
- Python, Gradio, PyTorch, Torchvision
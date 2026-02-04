# Multimodal RAG with Qwen3 Vision-Language Models

A multimodal Retrieval-Augmented Generation (RAG) system built on Qwen3 Vision-Language models. This project enables semantic search, document retrieval, and question-answering over documents containing both text and visual content.

## Overview

This system implements a complete multimodal RAG pipeline with three core stages:

1. **Embedding Generation** - Create vector embeddings from text, images, or videos using Qwen3-VL-Embedding-2B
2. **Semantic Search & Reranking** - Retrieve relevant documents and refine results using Qwen3-VL-Reranker-2B
3. **Multimodal QA** - Answer questions about retrieved content using Qwen3-VL-2B-Instruct

## Architecture

```
Input: PDF / Images / Text
         │
         ├─→ Convert PDF to page images
         │
         ├─→ [EMBEDDING] Qwen3-VL-Embedding-2B
         │   └─→ 2048-dimensional embeddings
         │         │
         └─→ User Query
             │
             ├─→ Embed query
             │
             ├─→ [SEMANTIC SEARCH] Similarity scoring
             │   └─→ Retrieve top-k candidates
             │
             ├─→ [RERANKING] Qwen3-VL-Reranker-2B
             │   └─→ Relevance scores
             │
             └─→ [QA] Qwen3-VL-2B-Instruct
                 └─→ Generated answer
```

## Features

- **Multimodal Input Support**: Process PDFs, text, images
- **PDF Document Processing**: Convert PDFs to images and embed each page semantically
- **High-Performance Inference**: vLLM integration for optimized GPU inference
- **Flexible Embedding**: Custom `Qwen3VLEmbedder` class with configurable parameters
- **Two-Stage Retrieval**: Semantic search followed by reranking
- **OpenAI-Compatible API**: Works with vLLM's OpenAI-compatible endpoint

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU
- Poppler (for PDF processing)

### Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt
```

### Install Poppler (for PDF to image conversion)

```bash
# Ubuntu/Debian
sudo apt-get install poppler-utils

# macOS
brew install poppler
```

## Usage

### Complete RAG Pipeline

See [multimodal_rag.ipynb](multimodal_rag.ipynb) for a complete example that:

1. Converts a PDF to page images
2. Embeds each page
3. Performs semantic search on a user query
4. Reranks results
5. Generates an answer using the top-ranked document


## Project Structure

```
Multimodal_Embeddings/
├── helpers.py                 # Helper functions
├── reranker_template.jinja    # Reranking prompt template
├── multimodal_rag.ipynb       # Complete RAG pipeline demo
├── requirements.txt           # Minimal dependencies
└── data/                      # Sample documents
```

## Models Used

| Model | Purpose | Parameters |
|-------|---------|------------|
| [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) | Multimodal embedding generation | 2B |
| [Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) | Result reranking | 2B |
| [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | Question answering | 2B |

## Acknowledgments

- [Qwen Team](https://github.com/QwenLM/Qwen3-VL-Embedding) for the Qwen3-VL models
- [vLLM](https://github.com/vllm-project/vllm) for high-performance inference
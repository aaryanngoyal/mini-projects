# mini-projects

A collection of small, self-contained projects — experiments, learning exercises, and prototypes.

## Projects

### [`digit_classification_mnist`](./digit_classification_mnist)
A starting experiment on the classic MNIST dataset for handwritten digit classification. Trains a simple feedforward neural network to classify handwritten digits with ~99.7% validation accuracy.

### [`travel-itinerary-planner`](./travel-itinerary-planner)
An AI-powered travel planning assistant built with Google's Gemini API. Given a destination, dates, and budget, it checks the weather, searches for hotels, and produces a structured itinerary using Gemini's function-calling and structured output features.

### [`virtual-canvas`](./virtual-canvas)
A hand-tracking virtual canvas project using OpenCV. Detects skin-colored regions in a webcam feed, identifies fingertips, and enables freehand drawing on a canvas with smoothing.

### [`PDF-Chatbot-with-Citations`](./PDF-Chatbot-with-Citations)
A Retrieval-Augmented Generation (RAG) application that ingests PDF documents, retrieves relevant chunks based on semantic and keyword search, and generates answers with precise citations back to the source document. Handles complex PDF layouts including tables, figures, and multi-level headings.

## Structure

```
mini-projects/
├── digit_classification_mnist/
├── travel-itinerary-planner/
├── virtual-canvas/
└── PDF-Chatbot-with-Citations/
```

Each project folder contains its own code and, where applicable, its own README with setup and usage instructions.

## License

MIT

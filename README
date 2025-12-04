# Earthquake vs Noise Classification Using Time-Series Waveforms

This project explores whether machine learning and deep learning models can tell apart real earthquake events from background noise using raw 3-channel seismic waveform data. Each waveform contains 6000 time samples digitized to 14 bits across three channels.

The goal is to build a full waveform‐processing pipeline and evaluate different models to see which can reliably detect earthquakes.

## Project Overview

The repository includes:

* A Jupyter notebook that walks through the full workflow

* A preprocessing function that filters and standardizes raw waveform data

* A baseline anomaly detection model using Isolation Forest

* A 1D CNN model that learns time-domain structures directly from the waveform

* Plots, metrics, and comparisons between models

The project was built to demonstrate practical data processing, ML pipeline design, and model comparison on real scientific data.

## Performance Summary

### Model Performance

| Model                             | Input Type                     | Accuracy |
|----------------------------------|---------------------------------|----------|
| **CNN (1D Convolutional Model)** | Structured waveform `(6000, 3)` | **96.17%** |
| **Isolation Forest (Flattened)** | Flattened waveform `(18000,)`   | **94.33%** |
| **Isolation Forest (Time Features)** | Extracted time-domain features | **94.83%** |


The Isolation Forest provides a rough separation but has high overlap between noise and earthquake scores

The CNN model performs significantly better by learning structural waveform patterns

Maintaining 3‐channel structure improves detection compared to flattening
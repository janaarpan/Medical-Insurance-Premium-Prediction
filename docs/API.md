# API Reference

- `GET /`: service welcome message and documentation location.
- `GET /health`: process/model health status and model version.
- `POST /predict`: accepts age, weight (kg), height (m), income_lpa, smoker, city, and occupation; returns predicted_category, confidence, and class_probabilities.

Interactive request/response schemas are available at `/docs` while the server is running.

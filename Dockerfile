# Use a slim Python base
FROM python:3.11-slim

# Set working dir
WORKDIR /app

# Copy source
COPY . .

# Install dependencies
RUN pip install fastapi uvicorn jinja2 requests

# Expose port 5000
EXPOSE 5000

# Run the app
CMD ["uvicorn", "workout_app.server:app", "--host", "0.0.0.0", "--port", "5000"]

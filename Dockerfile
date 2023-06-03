# Starting from Python 3.11 base image
FROM python:3.11

# Set a directory for the app
WORKDIR /usr/src/app

# Copy all the files to the container
COPY . .

# Update the base container install (optional)
RUN apt-get update -y && apt-get upgrade -y

# Install dependencies
RUN pip install --no-cache-dir pandas xgboost duckdb click pyarrow transformers ipykernel matplotlib seaborn openai scikit-learn rdkit selfies numpy

# Run the application:
CMD ["python", "./your-python-script.py"]
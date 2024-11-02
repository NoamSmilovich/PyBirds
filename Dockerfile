# Use an official conda image
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy environment file
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "PyBirds", "/bin/bash", "-c"]

# Copy application code
COPY . .

# Make sure the environment is activated
RUN echo "conda activate PyBirds" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Start the application
CMD ["conda", "run", "-n", "PyBirds", "gunicorn", "app:app", "--bind", "0.0.0.0:8000"]

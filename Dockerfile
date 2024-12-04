FROM python:3.11.2-slim
# Set the working directory
WORKDIR /app

ENV PORT 3500

# Copy the current directory contents into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 3500 available to the world outside this container
EXPOSE 3500

# Command to run the application
CMD ["python", "main.py"]
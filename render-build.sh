#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install system dependencies including audio libraries
apt-get update -y
apt-get install -y libasound2-dev ffmpeg libsndfile1 sox 

# Navigate to the backend directory
cd backend

# Install dependencies
npm install

# Build the app
npm run build

echo "Build script completed" 
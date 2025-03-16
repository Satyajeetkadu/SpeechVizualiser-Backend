# SpeechVizualiser Backend

Backend service for the SpeechVizualiser application, which analyzes speech patterns and provides insights.

## Features

- Speech transcription and analysis
- Silence comfort analysis
- Pitch visualization
- Language precision analysis
- Visual language analysis
- Erosion tags detection
- Vocal archetypes classification

## Setup

1. Install dependencies:
   ```
   npm install
   ```

2. Create a `.env` file based on `.env.example` and add your API keys.

3. Start the development server:
   ```
   npm run dev
   ```

## API Endpoints

- `POST /api/upload`: Upload an audio file for analysis
- `POST /api/upload/snippet`: Upload an audio snippet for analysis
- `GET /api/health`: Health check endpoint 
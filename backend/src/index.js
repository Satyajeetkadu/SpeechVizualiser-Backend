import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import OpenAI from 'openai';
import { ElevenLabsClient } from 'elevenlabs';
import { fileURLToPath } from 'url';
import ffmpeg from 'fluent-ffmpeg';
import Meyda from 'meyda';
import { Readable } from 'stream';
import { exec } from 'child_process';
import { promisify } from 'util';
import { AudioContext } from 'node-web-audio-api';

// Setup __dirname equivalent for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config();

const app = express();
const port = process.env.PORT || 8080;

// Initialize OpenAI
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Initialize ElevenLabs
const elevenlabs = new ElevenLabsClient({
  apiKey: process.env.ELEVENLABS_API_KEY
});

const execAsync = promisify(exec);

// CORS configuration
const corsOptions = {
  origin: process.env.CORS_ORIGIN ? process.env.CORS_ORIGIN.split(',') : ['http://localhost:5173'],
  credentials: true,
  optionsSuccessStatus: 200
};

app.use(cors(corsOptions));

// Add a middleware to log all incoming requests
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
  console.log('Headers:', JSON.stringify(req.headers, null, 2));
  next();
});

app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Ensure uploads directory exists
const uploadsDir = path.join(__dirname, '../uploads');
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Configure multer for audio file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadsDir);
  },
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, uniqueSuffix + path.extname(file.originalname));
  },
});

const upload = multer({
  storage: storage,
  limits: { 
    fileSize: 50 * 1024 * 1024, // 50MB limit (increased from 25MB)
    fieldSize: 50 * 1024 * 1024 // 50MB field size limit
  },
  fileFilter: (req, file, cb) => {
    console.log('Received file:', file.originalname, 'Mimetype:', file.mimetype);
    
    // Accept all audio file types
    if (file.mimetype.startsWith('audio/') || 
        file.originalname.endsWith('.mp3') || 
        file.originalname.endsWith('.wav') || 
        file.originalname.endsWith('.m4a') || 
        file.originalname.endsWith('.ogg')) {
      console.log('File accepted:', file.originalname);
      cb(null, true);
    } else {
      console.log('File rejected:', file.originalname, 'Mimetype:', file.mimetype);
      cb(new Error(`Only audio files are allowed. Received: ${file.mimetype}`));
    }
  },
});

// Serve static files from the uploads directory
app.use('/uploads', express.static(path.join(__dirname, '../uploads')));

// Add a simple endpoint to check if the audio snippet is accessible
app.get('/check-audio/:fileName', (req, res) => {
  const fileName = req.params.fileName;
  const filePath = path.join(__dirname, '../uploads/snippets', fileName);
  
  // Check if the file exists
  if (fs.existsSync(filePath)) {
    console.log(`Audio file exists: ${filePath}`);
    res.json({ 
      exists: true, 
      url: `/uploads/snippets/${fileName}`,
      fullPath: filePath
    });
  } else {
    console.log(`Audio file does not exist: ${filePath}`);
    res.status(404).json({ 
      exists: false, 
      url: `/uploads/snippets/${fileName}`,
      fullPath: filePath
    });
  }
});

// Helper functions
async function transcribeAudio(audioData) {
  try {
    console.log('\n=== Starting ElevenLabs Transcription ===');
    console.log('File path:', audioData.path);

    // Read the file as a buffer
    const fileBuffer = await fs.promises.readFile(audioData.path);
    
    // Create a Blob from the buffer
    const blob = new Blob([fileBuffer]);

    console.log('Initiating transcription request to ElevenLabs API...');
    const transcriptionResponse = await elevenlabs.speechToText.convert({
      file: blob,
      model_id: 'scribe_v1',
      language_code: 'en',
      timestamps_granularity: 'word',
      tag_audio_events: true,
      diarize: false
    });

    console.log('\n=== ElevenLabs API Response ===');
    console.log('Language:', transcriptionResponse.language_code);
    console.log('Language Probability:', transcriptionResponse.language_probability);
    console.log('Text:', transcriptionResponse.text);
    console.log('Words:', transcriptionResponse.words.length);

    // Format the response to match your existing structure
    const formattedResponse = {
      text: transcriptionResponse.text,
      words: transcriptionResponse.words.map(word => ({
        word: word.text,
        start: word.start,
        end: word.end,
        type: word.type
      })),
      language: transcriptionResponse.language_code,
      confidence: transcriptionResponse.language_probability
    };

    console.log('\n=== Formatted Response ===');
    console.log(JSON.stringify(formattedResponse, null, 2));
    console.log('\n=== Transcription Complete ===\n');

    return formattedResponse;
  } catch (error) {
    console.error('\n=== Transcription Error ===');
    console.error('Error:', error);
    throw new Error(`Failed to transcribe audio: ${error.message}`);
  }
}

// Add this function to extract audio snippet
async function extractAudioSnippet(audioFilePath) {
  try {
    // Create uploads/snippets directory if it doesn't exist
    const snippetsDir = path.join(__dirname, '../uploads/snippets');
    if (!fs.existsSync(snippetsDir)) {
      fs.mkdirSync(snippetsDir, { recursive: true });
    }
    
    // Get audio duration using ffprobe
    const durationCmd = `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${audioFilePath}"`;
    const { stdout: durationOutput } = await execAsync(durationCmd);
    const duration = parseFloat(durationOutput.trim());
    
    console.log(`Audio duration: ${duration} seconds`);
    
    if (isNaN(duration) || duration <= 0) {
      throw new Error('Could not determine audio duration');
    }
    
    // Calculate middle 15 seconds
    const snippetDuration = 15;
    let startTime = 0;
    
    if (duration > snippetDuration) {
      // Start at the middle minus half snippet duration
      startTime = Math.max(0, (duration / 2) - (snippetDuration / 2));
    }
    
    // Generate output filename
    const timestamp = Date.now();
    const audioExt = path.extname(audioFilePath);
    const snippetFileName = `snippet_${timestamp}${audioExt}`;
    const snippetFilePath = path.join(snippetsDir, snippetFileName);
    
    // Extract snippet using ffmpeg
    const ffmpegCmd = `ffmpeg -i "${audioFilePath}" -ss ${startTime} -t ${snippetDuration} -c copy "${snippetFilePath}"`;
    await execAsync(ffmpegCmd);
    
    console.log(`Audio snippet extracted to: ${snippetFilePath}`);
    
    // Return the URL to serve this file
    return `/uploads/snippets/${snippetFileName}`;
  } catch (error) {
    console.error('Error extracting audio snippet:', error);
    return null;
  }
}

// Add pitch analysis function
async function analyzePitch(audioPath, transcriptionData) {
  let processedAudioPath;
  try {
    console.log('\n=== Starting Pitch Analysis ===');
    console.log('Processing audio file:', audioPath);
    
    if (!audioPath) {
      throw new Error('No audio file path provided');
    }

    // Check if file exists
    if (!fs.existsSync(audioPath)) {
      throw new Error('Audio file not found');
    }

    // Convert audio to mono 16kHz WAV using ffmpeg
    processedAudioPath = path.join(
      path.dirname(audioPath),
      path.basename(audioPath, path.extname(audioPath)) + '_processed.wav'
    );

    console.log('Converting audio to WAV format...');
    await new Promise((resolve, reject) => {
      ffmpeg(audioPath)
        .toFormat('wav')
        .audioChannels(1)
        .audioFrequency(16000)
        .on('end', () => {
          console.log('Audio conversion complete');
          resolve();
        })
        .on('error', (err) => {
          console.error('FFmpeg error:', err);
          reject(err);
        })
        .save(processedAudioPath);
    });

    // Read the processed audio file
    const audioBuffer = await fs.promises.readFile(processedAudioPath);
    
    // Convert Buffer to ArrayBuffer
    const arrayBuffer = audioBuffer.buffer.slice(
      audioBuffer.byteOffset,
      audioBuffer.byteOffset + audioBuffer.byteLength
    );

    // Initialize Web Audio API context
    const audioContext = new AudioContext();
    
    // Decode audio data
    const audioData = await audioContext.decodeAudioData(arrayBuffer);
    
    // Configure Meyda analyzer
    Meyda.bufferSize = 512;
    Meyda.sampleRate = audioData.sampleRate;
    
    // Get audio data as Float32Array
    const channelData = audioData.getChannelData(0);
    
    // Process audio in chunks
    const frameSize = 512;
    const pitchData = [];
    
    for (let i = 0; i < channelData.length; i += frameSize) {
      const frame = channelData.slice(i, i + frameSize);
      if (frame.length === frameSize) {
        const features = Meyda.extract(['rms', 'perceptualSpread'], frame);
        pitchData.push({
          time: i / audioData.sampleRate,
          pitch: features.perceptualSpread * 100, // Scale to 0-100 range
          intensity: features.rms
        });
      }
    }

    // Process the pitch data into sentences
    const transcriptionWords = transcriptionData.words || [];
    const sentences = [];
    let currentSentence = {
      text: '',
      pitchPoints: [],
      upspeakIndices: []
    };
    
    for (const word of transcriptionWords) {
      if (word.type === 'word') {
        // Add word to current sentence
        currentSentence.text += word.word + ' ';
        
        // Find pitch points during this word
        const wordStartTime = parseFloat(word.start);
        const wordEndTime = parseFloat(word.end);
        
        const wordPitchPoints = pitchData.filter(p => 
          p.time >= wordStartTime && p.time <= wordEndTime
        );
        
        if (wordPitchPoints.length > 0) {
          // Add pitch points to the sentence
          currentSentence.pitchPoints.push(...wordPitchPoints);
          
          // Check for upspeak at the end of sentences
          if (word.word.endsWith('.') || word.word.endsWith('?') || word.word.endsWith('!')) {
            const lastPoints = wordPitchPoints.slice(-3);
            if (lastPoints.length >= 2 && 
                lastPoints[lastPoints.length - 1].pitch > lastPoints[0].pitch + 10) {
              currentSentence.upspeakIndices.push(currentSentence.pitchPoints.length - 1);
            }
            
            // Add sentence to list and start new one
            if (currentSentence.pitchPoints.length > 0) {
              sentences.push({
                ...currentSentence,
                text: currentSentence.text.trim()
              });
            }
            currentSentence = {
              text: '',
              pitchPoints: [],
              upspeakIndices: []
            };
          }
        }
      }
    }
    
    // Add any remaining sentence
    if (currentSentence.text && currentSentence.pitchPoints.length > 0) {
      sentences.push({
        ...currentSentence,
        text: currentSentence.text.trim()
      });
    }

    console.log('Pitch analysis complete. Found', sentences.length, 'sentences');
    console.log('Sample pitch data:', sentences[0]?.pitchPoints.slice(0, 3));
    
    // Extract audio snippet near the end of the function, before returning
    let snippetUrl = null;
    try {
      snippetUrl = await extractAudioSnippet(audioPath);
      console.log('Generated audio snippet URL:', snippetUrl);
    } catch (error) {
      console.error('Failed to extract audio snippet:', error);
    }
    
    // Add the audio URL to each sentence in the middle section
    if (snippetUrl) {
      // Find sentences in the middle third of the transcript
      const third = Math.floor(sentences.length / 3);
      const middleSection = sentences.slice(third, third * 2);
      
      // Add audio URL to middle section sentences
      middleSection.forEach(sentence => {
        sentence.audioUrl = snippetUrl;
      });
      
      console.log(`Added audio URL to ${middleSection.length} sentences in the middle section`);
    }
    
    return {
      sentences,
      pitchData,
      audioSnippetUrl: snippetUrl
    };

  } catch (error) {
    console.error('Error in pitch analysis:', error);
    // Return empty data structure
    return {
      sentences: [],
      pitchData: [],
      audioSnippetUrl: null
    };
  } finally {
    // Ensure we clean up the processed file if it exists
    try {
      if (processedAudioPath && fs.existsSync(processedAudioPath)) {
        await fs.promises.unlink(processedAudioPath);
      }
    } catch (error) {
      console.error('Error cleaning up processed file:', error);
    }
  }
}

function processPitchData(pitchData) {
  // Example sentences for visualization
  const sampleSentences = [
    "Hi, Joseph. Good morning. My name is Shakti, Shaktivel.",
    "I came across your Elite Speak Pro course or cohort program through your YouTube channel.",
    "I have overall 18-plus years of experience working in IT service organization."
  ];

  // Generate pitch visualization data for each sentence
  const sentenceData = sampleSentences.map((text, index) => {
    // Generate sample pitch points (we'll replace this with real data later)
    const numPoints = 20;
    const pitchPoints = Array(numPoints).fill(0).map((_, i) => {
      // Create a natural-looking pitch curve
      const base = 50; // Base pitch level
      const variation = 30; // Maximum variation
      const position = i / (numPoints - 1); // Position in the sentence (0 to 1)
      
      // Add some natural variation and a slight upward trend at the end
      const trend = position > 0.8 ? (position - 0.8) * 20 : 0; // Upward trend in last 20%
      const naturalVariation = Math.sin(position * 4) * 10; // Natural pitch variation
      
      return base + naturalVariation + trend;
    });

    // Identify potential upspeak by checking if the last few points show an upward trend
    const lastPoints = pitchPoints.slice(-5);
    const upspeakIndices = [];
    if (lastPoints[lastPoints.length - 1] > lastPoints[0]) {
      upspeakIndices.push(pitchPoints.length - 1);
    }

    return {
      text,
      pitchPoints,
      upspeakIndices
    };
  });

  return sentenceData;
}

// Add this function to analyze erosion tags
async function analyzeErosionTags(transcriptionText) {
  try {
    console.log('\n=== Starting Erosion Tags Analysis ===');
    
    // Call GPT to identify erosion tags
    const response = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [
        {
          role: "system",
          content: `You are an expert speech analyst specializing in identifying "erosion tags" - weak sentence endings that diminish the impact of speech.

OBJECTIVE:
Analyze a speech transcript to identify sentences that end with erosion tags.

WHAT ARE EROSION TAGS:
Erosion tags are weak sentence endings that include:
1. Filler words at the end of sentences (um, uh, like, you know)
2. Trailing off or incomplete thoughts ("and so...", "I mean...")
3. Unnecessary qualifiers ("sort of", "kind of", "I guess")
4. Redundant phrases ("and things like that", "and so forth")
5. Apologetic endings ("if that makes sense", "I hope that's clear")

ANALYSIS INSTRUCTIONS:
1. Identify up to 3 of the clearest examples of sentences with erosion tags
2. For each example, provide:
   - The complete sentence text
   - The exact point where the erosion tag begins (index)
   - The erosion tag text itself

OUTPUT FORMAT:
{
  "erosionTags": [
    {
      "sentence": "The complete sentence with the erosion tag",
      "ending": "the erosion tag part",
      "endingStart": number (index where the erosion tag begins in the sentence)
    }
  ]
}

EXAMPLES OF EROSION TAGS:
- "I think this project will be successful, um, you know."
- "We should focus on customer experience and stuff like that."
- "The data shows a clear trend, I guess."
- "This approach could work well, if that makes sense."
- "We need to improve our processes and, uh..."

IMPORTANT: Only include clear, definitive examples of erosion tags. Quality over quantity.`
        },
        {
          role: "user",
          content: transcriptionText
        }
      ],
      temperature: 0.7,
      max_tokens: 1000
    });

    console.log('\n=== Erosion Tags Analysis Response ===');
    console.log(response.choices[0].message.content);

    // Parse the response
    let erosionTagsData;
    try {
      erosionTagsData = JSON.parse(response.choices[0].message.content);
      console.log('\n=== Parsed Erosion Tags ===');
      console.log(JSON.stringify(erosionTagsData, null, 2));
    } catch (e) {
      console.error('Failed to parse erosion tags response:', e);
      // Provide empty array if parsing fails
      erosionTagsData = {
        erosionTags: []
      };
    }

    // Calculate a score based on the number of erosion tags found
    // Lower score is better (fewer erosion tags)
    const baseScore = 85; // Start with a good score
    const penaltyPerTag = 7; // Reduce score for each tag found
    
    // Calculate score (min 60, max 95)
    const score = Math.max(60, Math.min(95, 
      baseScore - (erosionTagsData.erosionTags?.length || 0) * penaltyPerTag
    ));
    
    console.log(`Erosion Tags Score: ${score}`);
    console.log('=== Erosion Tags Analysis Complete ===\n');

    return {
      sentences: erosionTagsData.erosionTags || [],
      score: score
    };
  } catch (error) {
    console.error('Error analyzing erosion tags:', error);
    return {
      sentences: [],
      score: 75 // Default score on error
    };
  }
}

// Add this function to analyze weak starters
async function analyzeWeakStarters(transcriptionText) {
  try {
    console.log('\n=== Starting Weak Starters Analysis ===');
    
    // Call GPT to identify weak starters
    const response = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [
        {
          role: "system",
          content: `You are an expert speech analyst specializing in identifying "weak starters" - phrases at the beginning of sentences that indicate uncertainty or lack of confidence.

OBJECTIVE:
Analyze a speech transcript to identify sentences that begin with weak starters.

WHAT ARE WEAK STARTERS:
Weak starters are phrases at the beginning of sentences that include:
1. Hedge phrases ("I think", "I guess", "I feel like", "Maybe", "Perhaps")
2. Apologetic openings ("I'm sorry but", "Excuse me", "I apologize")
3. Uncertainty markers ("I'm not sure if", "I don't know if")
4. Minimizers ("Just", "Only", "Simply")
5. Permission seekers ("If it's okay", "If you don't mind")

ANALYSIS INSTRUCTIONS:
1. Identify up to 3 of the clearest examples of sentences with weak starters
2. For each example, provide:
   - The complete sentence text
   - The weak starter phrase itself
   - The length of the weak starter phrase (number of characters)

OUTPUT FORMAT:
{
  "weakStarters": [
    {
      "sentence": "The complete sentence with the weak starter",
      "starter": "the weak starter phrase",
      "starterLength": number (length of the weak starter in characters)
    }
  ]
}

EXAMPLES OF WEAK STARTERS:
- "I think we should proceed with the project."
- "I'm not sure if this is the right approach."
- "Maybe we could try a different strategy."
- "I guess what I'm trying to say is..."
- "Just wanted to mention that..."

IMPORTANT: Only include clear, definitive examples of weak starters. Quality over quantity. Focus on the most impactful examples that demonstrate lack of confidence.`
        },
        {
          role: "user",
          content: transcriptionText
        }
      ],
      temperature: 0.7,
      max_tokens: 1000
    });

    console.log('\n=== Weak Starters Analysis Response ===');
    console.log(response.choices[0].message.content);

    // Parse the response
    let weakStartersData;
    try {
      weakStartersData = JSON.parse(response.choices[0].message.content);
      console.log('\n=== Parsed Weak Starters ===');
      console.log(JSON.stringify(weakStartersData, null, 2));
    } catch (e) {
      console.error('Failed to parse weak starters response:', e);
      // Provide empty array if parsing fails
      weakStartersData = {
        weakStarters: []
      };
    }

    // Calculate a score based on the number of weak starters found
    // Lower score is better (fewer weak starters)
    const baseScore = 90; // Start with a good score
    const penaltyPerStarter = 8; // Reduce score for each weak starter found
    
    // Calculate score (min 65, max 95)
    const score = Math.max(65, Math.min(95, 
      baseScore - (weakStartersData.weakStarters?.length || 0) * penaltyPerStarter
    ));
    
    // Count total hedge phrases in the transcript
    const hedgeWords = ["i think", "i guess", "i feel", "maybe", "perhaps", "sort of", "kind of", "just", "only", "simply"];
    let totalHedgeCount = 0;
    
    // Convert to lowercase for case-insensitive matching
    const lowerText = transcriptionText.toLowerCase();
    
    // Count occurrences of each hedge phrase
    hedgeWords.forEach(hedge => {
      let startPos = 0;
      while ((startPos = lowerText.indexOf(hedge, startPos)) !== -1) {
        totalHedgeCount++;
        startPos += hedge.length;
      }
    });
    
    console.log(`Weak Starters Score: ${score}`);
    console.log(`Total hedge phrases found: ${totalHedgeCount}`);
    console.log('=== Weak Starters Analysis Complete ===\n');

    return {
      sentences: weakStartersData.weakStarters || [],
      hedgePhrases: hedgeWords,
      confidenceScore: score,
      totalHedgeCount: totalHedgeCount
    };
  } catch (error) {
    console.error('Error analyzing weak starters:', error);
    return {
      sentences: [],
      hedgePhrases: [],
      confidenceScore: 80,
      totalHedgeCount: 0
    };
  }
}

// Add this function to analyze thought flow
async function analyzeThoughtFlow(transcriptionText) {
  try {
    console.log('\n=== Starting Thought Flow Analysis ===');
    
    // Call GPT to analyze the thought flow
    const response = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [
        {
          role: "system",
          content: `You are an expert speech analyst specializing in analyzing the flow of thought in speeches.

OBJECTIVE:
Analyze a speech transcript to identify key points/topics throughout the speech and their relevance to the main idea.

ANALYSIS INSTRUCTIONS:
1. Identify 10-15 key points or topics that appear throughout the speech
2. For each key point:
   - Determine its position in the speech (as a percentage from 0-100%)
   - Rate its relevance to the main idea/purpose of the speech (on a scale of 1-10)
   - Create a short label (2-4 words) that captures the essence of that point

OUTPUT FORMAT:
{
  "dataPoints": [
    { "progress": number (0-100), "relevance": number (1-10), "label": "short label" },
    ...more data points...
  ],
  "coherenceScore": number (0-100)
}

IMPORTANT NOTES:
- Ensure the points are distributed throughout the speech (from beginning to end)
- The relevance score should reflect how important each point is to the main message
- Higher relevance scores mean the point is more central to the main idea
- Coherence score reflects how well the speech maintains focus and logical flow

EXAMPLE:
For a job interview speech, data points might include:
{ "progress": 0, "relevance": 2, "label": "Greeting" }
{ "progress": 15, "relevance": 7, "label": "Background Experience" }
{ "progress": 40, "relevance": 9, "label": "Key Accomplishments" }
{ "progress": 90, "relevance": 8, "label": "Future Goals" }`
        },
        {
          role: "user",
          content: transcriptionText
        }
      ],
      temperature: 0.7,
      max_tokens: 2000
    });

    console.log('\n=== Thought Flow Analysis Response ===');
    console.log(response.choices[0].message.content);

    // Parse the response
    let thoughtFlowData;
    try {
      thoughtFlowData = JSON.parse(response.choices[0].message.content);
      console.log('\n=== Parsed Thought Flow Data ===');
      console.log(JSON.stringify(thoughtFlowData, null, 2));
    } catch (e) {
      console.error('Failed to parse thought flow response:', e);
      // Provide default data if parsing fails
      thoughtFlowData = {
        dataPoints: [
          { progress: 0, relevance: 2, label: "Greeting" },
          { progress: 20, relevance: 5, label: "Main Point" },
          { progress: 50, relevance: 7, label: "Key Insight" },
          { progress: 80, relevance: 6, label: "Conclusion" },
          { progress: 100, relevance: 3, label: "Closing" }
        ],
        coherenceScore: 75
      };
    }
    
    console.log('Thought Flow Analysis complete.');
    return thoughtFlowData;
  } catch (error) {
    console.error('Error analyzing thought flow:', error);
    // Return default data on error
    return {
      dataPoints: [
        { progress: 0, relevance: 2, label: "Greeting" },
        { progress: 20, relevance: 5, label: "Main Point" },
        { progress: 50, relevance: 7, label: "Key Insight" },
        { progress: 80, relevance: 6, label: "Conclusion" },
        { progress: 100, relevance: 3, label: "Closing" }
      ],
      coherenceScore: 75
    };
  }
}

// Function to analyze vocal archetypes based on transcript and pitch data
async function analyzeVocalArchetypes(transcript, pitchData) {
  console.log("Analyzing vocal archetypes...");
  
  try {
    // Extract relevant features from the pitch data
    const features = extractPitchFeatures(pitchData);
    
    // Use GPT to analyze the vocal archetype based on transcript and features
    const prompt = `
You are an expert speech analyst who specializes in identifying vocal archetypes in speech. 
I need you to analyze the following transcript and speech features and classify the speaker 
into these three specific archetypes: The Valiant, The Caregiver, and The Sage.

Transcript: "${transcript}"

Speech Features:
- Average Pitch: ${features.averagePitch} (Higher values indicate higher pitch)
- Pitch Variability: ${features.pitchVariability} (Higher values indicate more varied intonation)
- Speaking Rate: ${features.speakingRate} words per minute
- Pause Frequency: ${features.pauseFrequency} pauses per minute

For reference, here are the archetype definitions:

1. The Valiant:
   - Increased rate of speech
   - High level of energy and volume
   - Short and punchy delivery
   - Purposeful movement in speech patterns
   - Examples: Motivational speakers, religious sermons, Tony Robbins, Les Brown
   - Primary purpose: Move people to action

2. The Caregiver:
   - Slower rates of speech
   - Lower volume
   - Longer pauses that create connection
   - Radiates love, care & empathy
   - Warm, nurturing tone
   - Primary purpose: Connect emotionally and provide support

3. The Sage:
   - Slower rates of speech (especially when explaining complex topics)
   - More frequent pauses (helps audience digest content)
   - Matter-of-fact pitch pattern (sentences often end on lower pitch)
   - Thoughtful, measured delivery
   - Example: Engaging university lecturer
   - Primary purpose: Impart knowledge and wisdom

Based on the transcript and speech features, assign a percentage score (0-100) for each archetype, where the total should add up to approximately 100%.
Also determine the dominant archetype, and provide a brief analysis (2-3 sentences) explaining why.

Format your response exactly as follows, without any preamble or additional text:
{
  "archetypes": [
    { "name": "The Valiant", "score": 0, "color": "#FFC107" },
    { "name": "The Caregiver", "score": 0, "color": "#E91E63" },
    { "name": "The Sage", "score": 0, "color": "#2196F3" }
  ],
  "dominantArchetype": "",
  "analysis": ""
}
`;

    const completion = await openai.chat.completions.create({
      model: process.env.GPT_MODEL || "gpt-4",
      messages: [{ "role": "user", "content": prompt }],
      temperature: 0.7,
      max_tokens: 500
    });

    const content = completion.choices[0].message.content.trim();
    console.log("Vocal archetype analysis response:", content);
    
    try {
      // Extract the JSON part from the response
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      const jsonStr = jsonMatch ? jsonMatch[0] : null;
      
      if (!jsonStr) {
        console.error("No valid JSON found in GPT response");
        return getDefaultVocalArchetypes();
      }
      
      const result = JSON.parse(jsonStr);
      return result;
    } catch (jsonError) {
      console.error("Error parsing vocal archetype analysis JSON:", jsonError);
      return getDefaultVocalArchetypes();
    }
  } catch (error) {
    console.error("Error analyzing vocal archetypes:", error);
    return getDefaultVocalArchetypes();
  }
}

// Helper function to extract speech features from pitch data
function extractPitchFeatures(pitchData) {
  try {
    // Calculate average pitch
    const pitchValues = pitchData.flatMap(segment => segment.pitchValues.filter(p => p > 0));
    const averagePitch = pitchValues.length > 0 
      ? Math.round(pitchValues.reduce((sum, val) => sum + val, 0) / pitchValues.length)
      : 0;
    
    // Calculate pitch variability (standard deviation of pitch)
    let pitchVariability = 0;
    if (pitchValues.length > 0) {
      const squaredDiffs = pitchValues.map(val => Math.pow(val - averagePitch, 2));
      const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / pitchValues.length;
      pitchVariability = Math.round(Math.sqrt(variance));
    }
    
    // Calculate speaking rate (words per minute)
    const totalWords = pitchData.reduce((sum, segment) => sum + (segment.text.split(/\s+/).length || 0), 0);
    const totalDuration = pitchData.reduce((sum, segment) => sum + (segment.duration || 0), 0);
    const speakingRate = totalDuration > 0 
      ? Math.round((totalWords / totalDuration) * 60) 
      : 0;
    
    // Calculate pause frequency
    const pauseCount = pitchData.length > 1 ? pitchData.length - 1 : 0;
    const pauseFrequency = totalDuration > 0 
      ? Math.round((pauseCount / totalDuration) * 60) 
      : 0;
    
    return {
      averagePitch,
      pitchVariability,
      speakingRate,
      pauseFrequency
    };
  } catch (error) {
    console.error("Error extracting pitch features:", error);
    return {
      averagePitch: 0,
      pitchVariability: 0,
      speakingRate: 0,
      pauseFrequency: 0
    };
  }
}

// Default vocal archetypes in case of analysis failure
function getDefaultVocalArchetypes() {
  return {
    archetypes: [
      { name: "The Valiant", score: 30, color: "#FFC107" },
      { name: "The Caregiver", score: 35, color: "#E91E63" },
      { name: "The Sage", score: 35, color: "#2196F3" }
    ],
    dominantArchetype: "The Sage",
    analysis: "The speaker demonstrates a balanced communication style with elements of all three archetypes. There are some characteristics of The Sage in the measured delivery and thoughtful explanations."
  };
}

// Add this function to analyze visual language
async function analyzeVisualLanguage(transcriptionText) {
  try {
    console.log('\n=== Starting Visual Language Analysis ===');
    
    // Call GPT to analyze visual vs. functional language
    const response = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [
        {
          role: "system",
          content: `You are an AI linguistic analyst specializing in visual language detection.

OBJECTIVE:
Analyze a speech transcript to identify the balance between metaphoric (visual) speech and functional speech.

WHAT IS METAPHORIC/VISUAL SPEECH:
- Uses vivid imagery and figurative language
- Contains metaphors, similes, analogies
- Creates mental pictures through descriptive language
- Uses sensory details and evocative words
- Includes idioms and colorful expressions

WHAT IS FUNCTIONAL SPEECH:
- Direct, straightforward communication
- Factual statements and literal descriptions
- Procedural or instructional language
- Abstract concepts without imagery
- Plain, matter-of-fact expressions

ANALYSIS INSTRUCTIONS:
1. Carefully analyze the provided transcript
2. Identify sentences or phrases that use metaphoric/visual language
3. Calculate the approximate percentage of metaphoric vs. functional speech
4. Find the most notable examples of both types of speech (up to 3 each)

OUTPUT FORMAT:
{
  "breakdown": {
    "visual": number (percentage of visual language, 0-100),
    "functional": number (percentage of functional language, 0-100),
    "examples": {
      "visual": [
        "example visual language phrase 1",
        "example visual language phrase 2",
        "example visual language phrase 3"
      ],
      "functional": [
        "example functional language phrase 1",
        "example functional language phrase 2",
        "example functional language phrase 3"
      ]
    }
  },
  "score": number (0-100, representing the overall visual language score)
}

IMPORTANT: The percentages of visual and functional language must sum to 100%. The score should generally correlate with the percentage of visual language but can be adjusted based on the quality and impact of the visual language used.`
        },
        {
          role: "user",
          content: transcriptionText
        }
      ],
      temperature: 0.7,
      max_tokens: 1500
    });

    console.log('\n=== Visual Language Analysis Response ===');
    console.log(response.choices[0].message.content);

    // Parse the response
    let visualLanguageData;
    try {
      visualLanguageData = JSON.parse(response.choices[0].message.content);
      console.log('\n=== Parsed Visual Language Data ===');
      console.log(JSON.stringify(visualLanguageData, null, 2));
    } catch (e) {
      console.error('Failed to parse visual language response:', e);
      // Provide default data if parsing fails
      visualLanguageData = {
        breakdown: {
          visual: 40,
          functional: 60,
          examples: {
            visual: [],
            functional: []
          }
        },
        score: 65
      };
    }
    
    console.log('Visual Language Analysis complete.');
    return visualLanguageData;
  } catch (error) {
    console.error('Error analyzing visual language:', error);
    // Return default data on error
    return {
      breakdown: {
        visual: 40,
        functional: 60,
        examples: {
          visual: [],
          functional: []
        }
      },
      score: 65
    };
  }
}

// Add this function to analyze pronoun usage for anxiousness/insecurity
async function analyzePronounUsage(transcriptionText) {
  try {
    console.log('\n=== Starting Pronoun Usage Analysis ===');
    
    // Prepare the text by cleaning it
    const cleanText = transcriptionText.replace(/[^\w\s']/g, ' ').toLowerCase();
    const words = cleanText.split(/\s+/);
    
    // Initialize counters for different pronouns
    const pronounCounts = {
      i: 0,       // Self-focused (I, me, my, mine)
      we: 0,      // Inclusive (we, us, our)
      you: 0,     // Other-focused (you, your)
      other: 0    // Other words
    };
    
    // Define pronoun categories
    const selfPronouns = ['i', 'me', 'my', 'mine', 'myself'];
    const inclusivePronouns = ['we', 'us', 'our', 'ours', 'ourselves'];
    const otherPronouns = ['you', 'your', 'yours', 'yourself', 'yourselves'];
    
    // Count pronouns
    let totalWords = 0;
    let totalPronouns = 0;
    
    words.forEach(word => {
      if (word.length > 0) {
        totalWords++;
        
        if (selfPronouns.includes(word)) {
          pronounCounts.i++;
          totalPronouns++;
        } else if (inclusivePronouns.includes(word)) {
          pronounCounts.we++;
          totalPronouns++;
        } else if (otherPronouns.includes(word)) {
          pronounCounts.you++;
          totalPronouns++;
        }
      }
    });
    
    // Calculate percentages
    const totalAnalyzedPronouns = pronounCounts.i + pronounCounts.we + pronounCounts.you;
    
    const distribution = {
      self: totalAnalyzedPronouns > 0 ? Math.round((pronounCounts.i / totalAnalyzedPronouns) * 100) : 0,
      inclusive: totalAnalyzedPronouns > 0 ? Math.round((pronounCounts.we / totalAnalyzedPronouns) * 100) : 0,
      other: totalAnalyzedPronouns > 0 ? Math.round((pronounCounts.you / totalAnalyzedPronouns) * 100) : 0
    };
    
    // Ensure percentages add up to 100%
    const sum = distribution.self + distribution.inclusive + distribution.other;
    if (sum !== 100 && totalAnalyzedPronouns > 0) {
      // Adjust the largest value to make sum 100
      if (distribution.self >= distribution.inclusive && distribution.self >= distribution.other) {
        distribution.self += (100 - sum);
      } else if (distribution.inclusive >= distribution.self && distribution.inclusive >= distribution.other) {
        distribution.inclusive += (100 - sum);
} else {
        distribution.other += (100 - sum);
      }
    }
    
    // Calculate anxiousness score (lower is better)
    // High self-focus (I, me, my) can indicate anxiousness/insecurity
    // A healthy balance would have more inclusive pronouns
    let anxiousnessScore = Math.max(30, Math.min(100, 100 - distribution.self));
    
    // Find example sentences for each pronoun type
    const sentences = transcriptionText.split(/[.!?]+/).filter(s => s.trim().length > 0);
    
    const examples = {
      self: [],
      inclusive: [],
      other: []
    };
    
    // Helper function to check if a sentence contains specific pronouns
    const containsPronoun = (sentence, pronounList) => {
      const words = sentence.toLowerCase().split(/\s+/);
      return words.some(word => {
        const cleanWord = word.replace(/[^\w']/g, '');
        return pronounList.includes(cleanWord);
      });
    };
    
    // Find up to 3 examples for each pronoun type
    sentences.forEach(sentence => {
      const trimmed = sentence.trim();
      if (trimmed.length === 0) return;
      
      if (examples.self.length < 3 && containsPronoun(trimmed, selfPronouns)) {
        examples.self.push(trimmed);
      }
      
      if (examples.inclusive.length < 3 && containsPronoun(trimmed, inclusivePronouns)) {
        examples.inclusive.push(trimmed);
      }
      
      if (examples.other.length < 3 && containsPronoun(trimmed, otherPronouns)) {
        examples.other.push(trimmed);
      }
    });
    
    console.log('\n=== Pronoun Usage Analysis Results ===');
    console.log('Pronoun Counts:', pronounCounts);
    console.log('Distribution:', distribution);
    console.log('Anxiousness Score:', anxiousnessScore);
    console.log('Examples Found:', {
      self: examples.self.length,
      inclusive: examples.inclusive.length,
      other: examples.other.length
    });
    
    return {
      counts: pronounCounts,
      distribution: distribution,
      examples: examples,
      anxiousnessScore: anxiousnessScore,
      totalPronouns: totalPronouns,
      totalWords: totalWords
    };
  } catch (error) {
    console.error('Error analyzing pronoun usage:', error);
    // Return default data on error
    return {
      counts: { i: 0, we: 0, you: 0, other: 0 },
      distribution: { self: 33, inclusive: 33, other: 34 },
      examples: { self: [], inclusive: [], other: [] },
      anxiousnessScore: 70,
      totalPronouns: 0,
      totalWords: 0
    };
  }
}

// Modify the analyzeTranscript function to include vocal archetypes analysis
async function analyzeTranscript(transcriptionData) {
  try {
    console.log('\n=== Starting Language Precision Analysis ===');
    console.log('Analyzing transcript:', transcriptionData.text);
    
    const transcriptionText = transcriptionData.text;

    // Get AI analysis of the speech patterns
    const response = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [
        {
          role: "system",
          content: `You are an expert linguistic analyst specializing in language precision detection.

OBJECTIVE:
Analyze speech transcripts to identify very short, impactful phrases (2-5 words) that demonstrate precise language use.

ANALYSIS CRITERIA:
1. Identify short, impactful phrases (2-5 words) that:
   - Demonstrate technical or domain expertise (e.g., "go through parataxis")
   - Show intentional and sophisticated word choice (e.g., "natural way of speaking")
   - Create vivid or specific descriptions (e.g., "good, plain English")
   - Express complex ideas concisely (e.g., "don't know cattle")

2. STRICT EXCLUSIONS:
   - Filler words (um, uh, like, actually, basically, etc.)
   - Common prepositions or articles alone
   - Vague or general phrases
   - Hedging language (sort of, kind of, maybe, etc.)

3. Phrase Selection Rules:
   - Focus on VERY SHORT phrases (2-5 words) that are impactful
   - Each phrase should be a distinct concept
   - Phrases should start and end at word boundaries
   - Prefer phrases that stand out as particularly precise, technical, or vivid

4. Before returning any phrase:
   - Verify it's a short, impactful phrase (not a full sentence)
   - Check that it contains no filler words
   - Ensure it represents a complete concept
   - Validate the start and end indices are at word boundaries

OUTPUT FORMAT:
{
  "preciseLanguage": [
    {
      "phrase": "short impactful phrase",
      "significance": "explanation of why this demonstrates precise language",
      "startIndex": number (index where the phrase starts in the full text),
      "endIndex": number (index where the phrase ends in the full text),
      "category": "technical|descriptive|conceptual|analytical"
    }
  ]
}

EXAMPLES OF GOOD PHRASES:
- "go through parataxis"
- "good, plain English"
- "don't know cattle"
- "the best in England"
- "natural way of speaking"`
        },
        {
          role: "user",
          content: transcriptionText
        }
      ],
      temperature: 0.7,
      max_tokens: 2000
    });

    console.log('\n=== AI Analysis Response ===');
    console.log(response.choices[0].message.content);

    // Parse the AI analysis
    let parsedAnalysis;
    try {
      parsedAnalysis = JSON.parse(response.choices[0].message.content);
      
      // Log detailed analysis
      console.log('\n=== Precise Language Breakdown ===');
      console.log('Precise Phrases:', parsedAnalysis.preciseLanguage.length);
      console.log('Phrases:', parsedAnalysis.preciseLanguage);
      
      // Validate and fix the phrases
      parsedAnalysis.preciseLanguage = (parsedAnalysis.preciseLanguage || []).map(phrase => {
        // Verify the phrase text matches the indices in the transcript
        const actualText = transcriptionText.substring(phrase.startIndex, phrase.endIndex);
        if (actualText !== phrase.phrase) {
          console.log(`\nPhrase text mismatch:`);
          console.log(`Expected: "${phrase.phrase}"`);
          console.log(`Actual from indices: "${actualText}"`);
          
          // Try to find the correct indices
          const phraseIndex = transcriptionText.indexOf(phrase.phrase);
          if (phraseIndex !== -1) {
            console.log(`Found phrase at index ${phraseIndex}`);
            phrase.startIndex = phraseIndex;
            phrase.endIndex = phraseIndex + phrase.phrase.length;
          } else {
            console.log(`Could not find exact phrase in transcript`);
          }
        }
        
        // Verify the phrase is actually in the transcript at the specified indices
        console.log(`\nVerifying phrase: "${phrase.phrase}"`);
        console.log(`At indices: ${phrase.startIndex}-${phrase.endIndex}`);
        console.log(`Text at indices: "${transcriptionText.substring(phrase.startIndex, phrase.endIndex)}"`);
        
        // Log the significance data for debugging
        console.log(`\nPhrase: "${phrase.phrase}"`);
        console.log(`Significance: "${phrase.significance}"`);
        console.log(`Category: "${phrase.category}"`);
        console.log(`In sentence: "${transcriptionText.substring(phrase.startIndex, phrase.endIndex)}"`);
        
        // If the extracted text doesn't match the phrase, try to find the exact phrase in the sentence
        if (actualText !== phrase.phrase) {
          console.log(`Text mismatch! Searching for exact phrase in sentence...`);
          
          // Try to find the exact phrase in the sentence
          const phraseIndex = transcriptionText.indexOf(phrase.phrase);
          if (phraseIndex !== -1) {
            phrase.startIndex = phraseIndex;
            phrase.endIndex = phraseIndex + phrase.phrase.length;
            console.log(`Found exact phrase at indices ${phrase.startIndex}-${phrase.endIndex}`);
            console.log(`New extracted text: "${transcriptionText.substring(phrase.startIndex, phrase.endIndex)}"`);
          } else {
            // If exact phrase not found, try case-insensitive search
            const lowerSentence = transcriptionText.toLowerCase();
            const lowerPhrase = phrase.phrase.toLowerCase();
            const lowerPhraseIndex = lowerSentence.indexOf(lowerPhrase);
            
            if (lowerPhraseIndex !== -1) {
              phrase.startIndex = lowerPhraseIndex;
              phrase.endIndex = lowerPhraseIndex + phrase.phrase.length;
              console.log(`Found phrase (case-insensitive) at indices ${phrase.startIndex}-${phrase.endIndex}`);
              console.log(`New extracted text: "${transcriptionText.substring(phrase.startIndex, phrase.endIndex)}"`);
            } else {
              console.log(`Could not find exact phrase in sentence, using original indices`);
            }
          }
        }

        return phrase;
      });
      
    } catch (e) {
      console.error('Failed to parse AI response as JSON:', e);
      // Provide empty preciseLanguage array if parsing fails
      parsedAnalysis = {
        preciseLanguage: []
      };
    }
    
    // Calculate language precision score based on the number of precise phrases
    const precisionScore = Math.min(100, 
      Math.max(70, 
        // Base score of 70, plus 5 points per precise phrase
        70 + (parsedAnalysis.preciseLanguage.length * 5)
      )
    );

    // Add pitch analysis only if audioPath is available
    let pitchAnalysis = {
      sentences: [] // Default empty array if pitch analysis fails
    };
    
    if (transcriptionData.path) {
      try {
        pitchAnalysis = await analyzePitch(transcriptionData.path, transcriptionData);
      } catch (error) {
        console.error('Pitch analysis error:', error);
        // Continue with empty pitch analysis rather than failing completely
      }
    }

    // Add weak starters analysis
    const weakStartersAnalysis = await analyzeWeakStarters(transcriptionText);
    
    // Add thought flow analysis
    const thoughtFlowAnalysis = await analyzeThoughtFlow(transcriptionText);
    
    // Add vocal archetypes analysis
    const vocalArchetypesAnalysis = await analyzeVocalArchetypes(transcriptionText, pitchAnalysis);
    
    // Add visual language analysis
    const visualLanguageAnalysis = await analyzeVisualLanguage(transcriptionText);
    
    // Add pronoun usage analysis
    const pronounUsageAnalysis = await analyzePronounUsage(transcriptionText);
    
    const insights = {
      languagePrecision: {
        preciseInstances: (() => {
          // Log the raw phrases from the analysis
          console.log('\n=== Raw Precise Phrases ===');
          console.log(JSON.stringify(parsedAnalysis.preciseLanguage || [], null, 2));
          
          // Map the phrases directly to the format expected by the frontend
          const instances = (parsedAnalysis.preciseLanguage || [])
            .slice(0, 7)
            .map(phrase => {
              // Find the complete sentence containing this phrase for context
              let sentenceStart = Math.max(0, 
                transcriptionText.lastIndexOf('.', phrase.startIndex) + 1
              );
              if (sentenceStart === 0) {
                // If no period found, try looking for other sentence boundaries
                const altStart = Math.max(0,
                  Math.max(
                    transcriptionText.lastIndexOf('?', phrase.startIndex) + 1,
                    transcriptionText.lastIndexOf('!', phrase.startIndex) + 1
                  )
                );
                if (altStart > 0) sentenceStart = altStart;
              }
              
              let sentenceEnd = transcriptionText.indexOf('.', phrase.endIndex);
              if (sentenceEnd === -1) {
                // If no period found, try looking for other sentence boundaries
                sentenceEnd = Math.max(
                  transcriptionText.indexOf('?', phrase.endIndex),
                  transcriptionText.indexOf('!', phrase.endIndex)
                );
                // If still not found, use the end of text
                if (sentenceEnd === -1) sentenceEnd = transcriptionText.length;
              } else {
                sentenceEnd += 1; // Include the period
              }
              
              const fullSentence = transcriptionText.substring(sentenceStart, sentenceEnd).trim();

              // Calculate the relative position of the highlight within the sentence
              let relativeStart = phrase.startIndex - sentenceStart;
              let relativeEnd = phrase.endIndex - sentenceStart;

              // Verify the phrase text matches what's in the sentence at the calculated positions
              const extractedText = fullSentence.substring(relativeStart, relativeEnd);
              console.log(`\nPhrase: "${phrase.phrase}"`);
              console.log(`In sentence: "${fullSentence}"`);
              console.log(`Original indices: ${phrase.startIndex}-${phrase.endIndex}`);
              console.log(`Relative indices: ${relativeStart}-${relativeEnd}`);
              console.log(`Extracted text: "${extractedText}"`);
              
              // If the extracted text doesn't match the phrase, try to find the exact phrase in the sentence
              if (extractedText !== phrase.phrase) {
                console.log(`Text mismatch! Searching for exact phrase in sentence...`);
                
                // Try to find the exact phrase in the sentence
                const phraseIndex = fullSentence.indexOf(phrase.phrase);
                if (phraseIndex !== -1) {
                  relativeStart = phraseIndex;
                  relativeEnd = phraseIndex + phrase.phrase.length;
                  console.log(`Found exact phrase at indices ${relativeStart}-${relativeEnd}`);
                  console.log(`New extracted text: "${fullSentence.substring(relativeStart, relativeEnd)}"`);
                } else {
                  // If exact phrase not found, try case-insensitive search
                  const lowerSentence = fullSentence.toLowerCase();
                  const lowerPhrase = phrase.phrase.toLowerCase();
                  const lowerPhraseIndex = lowerSentence.indexOf(lowerPhrase);
                  
                  if (lowerPhraseIndex !== -1) {
                    relativeStart = lowerPhraseIndex;
                    relativeEnd = lowerPhraseIndex + phrase.phrase.length;
                    console.log(`Found phrase (case-insensitive) at indices ${relativeStart}-${relativeEnd}`);
                    console.log(`New extracted text: "${fullSentence.substring(relativeStart, relativeEnd)}"`);
                  } else {
                    console.log(`Could not find exact phrase in sentence, using original indices`);
                  }
                }
              }

              const highlightData = {
                text: phrase.phrase,
                start: relativeStart,
                end: relativeEnd,
                significance: phrase.significance,
                category: phrase.category
              };

              console.log('Highlight data being sent to frontend:', highlightData);
              
              return {
                sentence: fullSentence,
                highlights: [highlightData]
              };
            });

          // Log the final processed instances
          console.log('\n=== Final Processed Instances ===');
          console.log(JSON.stringify(instances, null, 2));
          
          return instances;
        })(),
        score: Math.min(100, 
          Math.max(70, 
            // Base score of 70, plus 5 points per precise phrase
            70 + (parsedAnalysis.preciseLanguage?.length || 0) * 5
          )
        )
      },
      pitchAnalysis: pitchAnalysis,
      silenceComfort: {
        groups: (() => {
          try {
            // Log initial transcription data
            console.log('\n=== Processing Silence Comfort Data ===');
            console.log('Initial words data:', transcriptionData.words?.slice(0, 2));

            // Ensure we have valid words data
            if (!Array.isArray(transcriptionData?.words) || transcriptionData.words.length === 0) {
              console.log('No valid words data found in transcription');
              return [];
            }

            // First, group words into sentences
            const sentences = [];
            let currentSentence = {
              text: '',
              words: [],
              start: null,
              end: null
            };

            // Process words into sentences with safety checks
            transcriptionData.words.forEach((word, idx) => {
              // Log first few words for debugging
              if (idx < 3) {
                console.log(`Processing word ${idx}:`, word);
              }

              // Skip if word is invalid
              if (!word || typeof word !== 'object') {
                console.log('Invalid word object:', word);
                return;
              }

              // Process valid word
              if (word.type === 'word') {
                // Use word.word instead of word.text
                const wordText = word.word || '';
                currentSentence.text += wordText + ' ';
                currentSentence.words.push(word);
                
                // Set start time if not set
                if (currentSentence.start === null) {
                  currentSentence.start = parseFloat(word.start || 0);
                }
                currentSentence.end = parseFloat(word.end || 0);

                // Check for sentence endings
                if (wordText.endsWith('.') || wordText.endsWith('!') || wordText.endsWith('?')) {
                  if (currentSentence.text.trim()) {
                    sentences.push({
                      ...currentSentence,
                      text: currentSentence.text.trim()
                    });
                    console.log('Added sentence:', currentSentence.text.trim());
                  }
                  currentSentence = {
                    text: '',
                    words: [],
                    start: null,
                    end: null
                  };
                }
              }
            });

            // Add any remaining sentence
            if (currentSentence.text.trim()) {
              sentences.push({
                ...currentSentence,
                text: currentSentence.text.trim()
              });
              console.log('Added final sentence:', currentSentence.text.trim());
            }

            // Log found sentences
            console.log('\nFound sentences:', sentences.length);
            if (sentences.length > 0) {
              console.log('First sentence:', sentences[0]);
            }

            // If no sentences were found, return empty array
            if (sentences.length === 0) {
              console.log('No valid sentences found in transcription');
              return [];
            }

            // Calculate pauses between all sentences
            const allPauses = [];
            for (let i = 0; i < sentences.length - 1; i++) {
              const currentSentence = sentences[i];
              const nextSentence = sentences[i + 1];
              const pauseDuration = nextSentence.start - currentSentence.end;

              if (pauseDuration > 0.2) {
                allPauses.push({
                  index: i,
                  start: currentSentence.end,
                  end: nextSentence.start,
                  duration: pauseDuration,
                  beforeSentence: currentSentence,
                  afterSentence: nextSentence
                });
              }
            }

            console.log(`\nFound ${allPauses.length} pauses in the transcript`);
            
            // Sort pauses by duration (descending) to find the most significant ones
            allPauses.sort((a, b) => b.duration - a.duration);
            
            // Get the top 3 pauses (or fewer if there aren't 3)
            const topPauses = allPauses.slice(0, Math.min(3, allPauses.length));
            
            console.log(`\nSelected top ${topPauses.length} pauses:`);
            topPauses.forEach((pause, i) => {
              console.log(`Pause ${i+1}: ${pause.duration.toFixed(2)}s between "${pause.beforeSentence.text.substring(0, 30)}..." and "${pause.afterSentence.text.substring(0, 30)}..."`);
            });
            
            // Create sections for each significant pause
            const sections = topPauses.map((pause, index) => {
              // Include the sentence before and after the pause
              const sectionSentences = [pause.beforeSentence, pause.afterSentence];
              
              // Create a section with a descriptive name based on the pause duration
              const sectionName = `pause-${index+1}`;
              const pauseDescription = pause.duration < 0.5 ? 'short' : 
                                      pause.duration > 2.0 ? 'long' : 'medium';
              
              return {
                section: sectionName,
                sectionTitle: `${pauseDescription.charAt(0).toUpperCase() + pauseDescription.slice(1)} Pause (${pause.duration.toFixed(1)}s)`,
                sentences: sectionSentences,
                pauses: [{
                  start: pause.start,
                  end: pause.end,
                  duration: pause.duration
                }]
              };
            });

            console.log('\n=== Silence Comfort Processing Complete ===\n');
            return sections;

          } catch (error) {
            console.error('Error processing silence comfort data:', error);
            return [];
          }
        })(),
        overallScore: (() => {
          try {
            // Calculate a real comfort score based on pause patterns
            const words = transcriptionData.words || [];
            if (words.length === 0) return 75; // Default score
            
            // Count pauses and calculate their durations
            let totalPauses = 0;
            let idealPauses = 0; // Pauses between 0.5 and 2.0 seconds
            let previousWordEnd = null;
            
            words.forEach(word => {
              if (word.type === 'word') {
                const wordStart = parseFloat(word.start || 0);
                
                if (previousWordEnd !== null) {
                  const pauseDuration = wordStart - previousWordEnd;
                  if (pauseDuration > 0.2) { // Minimum pause threshold
                    totalPauses++;
                    if (pauseDuration >= 0.5 && pauseDuration <= 2.0) {
                      idealPauses++;
                    }
                  }
                }
                
                previousWordEnd = parseFloat(word.end || 0);
              }
            });
            
            // Calculate score based on percentage of ideal pauses
            const score = totalPauses > 0 
              ? Math.round((idealPauses / totalPauses) * 100)
              : 75; // Default if no pauses found
              
            console.log(`\nSilence Comfort Score: ${score}% (${idealPauses} ideal pauses out of ${totalPauses} total)`);
            
            return score;
          } catch (error) {
            console.error('Error calculating silence comfort score:', error);
            return 75; // Default score on error
          }
        })()
      },
      erosionTags: await analyzeErosionTags(transcriptionText),
      weakStarters: weakStartersAnalysis,
      treeOfThought: {
        dataPoints: thoughtFlowAnalysis.dataPoints,
        coherenceScore: thoughtFlowAnalysis.coherenceScore
      },
      vocalArchetypes: vocalArchetypesAnalysis,
      visualLanguage: visualLanguageAnalysis,
      pronounUsage: pronounUsageAnalysis
    };

    console.log('\n=== Final Language Precision Score ===');
    console.log('Score:', Math.round(precisionScore));
    console.log('\n=== Analysis Complete ===\n');

    return insights;
  } catch (error) {
    console.error('\n=== Analysis Error ===');
    console.error('Error:', error);
    throw new Error('Failed to analyze transcription');
  }
}

// Routes
app.post('/api/analyze', upload.single('audio'), async (req, res) => {
  try {
    console.log('\n=== Received Audio Upload Request ===');
    console.log('Request body:', req.body);
    console.log('Request file:', req.file);
    
    if (!req.file) {
      console.log('Error: No audio file provided');
      return res.status(400).json({ error: 'No audio file provided' });
    }

    console.log(`\n=== Processing Audio File ===`);
    console.log(`Filename: ${req.file.originalname}`);
    console.log(`Size: ${req.file.size} bytes`);
    console.log(`Path: ${req.file.path}`);

    try {
      // Transcribe the audio
      const transcriptionData = await transcribeAudio(req.file);
      
      // Analyze the transcript
      const analysisData = await analyzeTranscript(transcriptionData);
      
      // Analyze pitch
      const pitchData = await analyzePitch(req.file.path, transcriptionData);
      
      console.log(`\n=== Analysis Results Structure ===`);
      console.log(`Pitch Data Structure:`);
      console.log(`- sentences: Array with ${pitchData.sentences?.length || 0} items`);
      console.log(`- audioSnippetUrl: ${pitchData.audioSnippetUrl || 'Not available'}`);
      
      // Build the response with all analysis data
      const responseData = {
        transcription: transcriptionData,
        analysis: {
          languagePrecision: analysisData.languagePrecision,
          pitch: pitchData.sentences,
          audioSnippetUrl: pitchData.audioSnippetUrl,
          silenceComfort: analysisData.silenceComfort,
          erosionTags: analysisData.erosionTags,
          weakStarters: analysisData.weakStarters,
          treeOfThought: analysisData.treeOfThought,
          vocalArchetypes: analysisData.vocalArchetypes,
          visualLanguage: analysisData.visualLanguage,
          pronounUsage: analysisData.pronounUsage
        }
      };
      
      console.log(`\n=== Response Data Structure ===`);
      console.log(`- transcription: Present`);
      console.log(`- analysis.languagePrecision: Present`);
      console.log(`- analysis.pitch: Array with ${responseData.analysis.pitch?.length || 0} items`);
      console.log(`- analysis.audioSnippetUrl: ${responseData.analysis.audioSnippetUrl || 'Not available'}`);
      console.log(`- analysis.silenceComfort: ${responseData.analysis.silenceComfort ? 'Present' : 'Missing'}`);
      
      // Log the full response data structure
      console.log(`\n=== Full Response Data (First 500 chars) ===`);
      console.log(JSON.stringify(responseData).substring(0, 500) + '...');
      
      // Log the transcription structure
      console.log(`\n=== Transcription Structure ===`);
      console.log(`- text: ${responseData.transcription.text ? 'Present' : 'Missing'}`);
      console.log(`- words: Array with ${responseData.transcription.words?.length || 0} items`);
      
      // Log the analysis structure
      console.log(`\n=== Analysis Structure ===`);
      console.log(`- languagePrecision: ${responseData.analysis.languagePrecision ? 'Present' : 'Missing'}`);
      console.log(`- pitch: ${responseData.analysis.pitch ? 'Present' : 'Missing'}`);
      console.log(`- silenceComfort: ${responseData.analysis.silenceComfort ? 'Present' : 'Missing'}`);
      
      // Log silence comfort data structure
      if (responseData.analysis.silenceComfort) {
        console.log(`\n=== Silence Comfort Data Structure ===`);
        console.log(`- overallScore: ${responseData.analysis.silenceComfort.overallScore || 0}`);
        console.log(`- groups: Array with ${responseData.analysis.silenceComfort.groups?.length || 0} sections`);
        
        if (responseData.analysis.silenceComfort.groups && responseData.analysis.silenceComfort.groups.length > 0) {
          console.log(`  - First section: ${responseData.analysis.silenceComfort.groups[0].section}`);
          console.log(`  - Sentences: ${responseData.analysis.silenceComfort.groups[0].sentences?.length || 0}`);
          console.log(`  - Pauses: ${responseData.analysis.silenceComfort.groups[0].pauses?.length || 0}`);
        }
      }

      // Return the response with a consistent structure
      res.json({
        success: true,
        data: responseData
      });
    } catch (innerError) {
      console.error('Error processing audio:', innerError);
      res.status(500).json({ 
        success: false,
        error: 'Error processing audio', 
        details: innerError.message 
      });
    }
  } catch (error) {
    console.error('Error handling request:', error);
    res.status(500).json({ 
      success: false,
      error: 'Error processing audio', 
      details: error.message 
    });
  }
});

// Route for handling audio snippets
app.post('/api/speech/analyze-snippet', upload.single('audio'), async (req, res) => {
  try {
    if (!req.file) {
      res.status(400).json({ error: 'No audio file provided' });
      return;
    }

    // Log snippet information
    console.log('\n=== Processing Audio Snippet ===');
    console.log('Start time:', req.body.startTime);
    console.log('End time:', req.body.endTime);
    console.log('File path:', req.file.path);
    console.log('File size:', req.file.size, 'bytes');

    // Get transcription with word-level timestamps
    const transcription = await transcribeAudio({
      path: req.file.path,
      isSnippet: true
    });

    // Add metadata to transcription
    transcription.path = req.file.path;
    transcription.isSnippet = true;
    transcription.startTime = parseFloat(req.body.startTime) || 0;
    transcription.endTime = parseFloat(req.body.endTime) || 0;

    // Analyze the transcription
    const analysis = await analyzeTranscript(transcription);
    
    // Analyze pitch - add this step
    const pitchData = await analyzePitch(req.file.path, transcription);
    
    console.log(`\n=== Snippet Analysis Results Structure ===`);
    console.log(`Pitch Data Structure:`);
    console.log(`- sentences: Array with ${pitchData.sentences?.length || 0} items`);
    console.log(`- audioSnippetUrl: ${pitchData.audioSnippetUrl || 'Not available'}`);
    
    // Build the complete response with all analysis data
    const responseData = {
      transcription,
      analysis: {
        languagePrecision: analysis.languagePrecision,
        pitch: pitchData.sentences,
        audioSnippetUrl: pitchData.audioSnippetUrl,
        silenceComfort: analysis.silenceComfort,
        erosionTags: analysis.erosionTags,
        weakStarters: analysis.weakStarters,
        treeOfThought: analysis.treeOfThought,
        vocalArchetypes: analysis.vocalArchetypes,
        visualLanguage: analysis.visualLanguage,
        pronounUsage: analysis.pronounUsage
      }
    };
    
    console.log(`\n=== Snippet Response Data Structure ===`);
    console.log(`- transcription: Present`);
    console.log(`- analysis.languagePrecision: ${responseData.analysis.languagePrecision ? 'Present' : 'Missing'}`);
    console.log(`- analysis.pitch: Array with ${responseData.analysis.pitch?.length || 0} items`);
    console.log(`- analysis.audioSnippetUrl: ${responseData.analysis.audioSnippetUrl || 'Not available'}`);
    console.log(`- analysis.silenceComfort: ${responseData.analysis.silenceComfort ? 'Present' : 'Missing'}`);

    // Clean up the temporary file after analysis is complete
    try {
      await fs.promises.unlink(req.file.path);
      console.log('Temporary file cleaned up:', req.file.path);
    } catch (cleanupError) {
      console.error('Error cleaning up temporary file:', cleanupError);
    }

    // Return the complete response
    res.json({
      success: true,
      data: responseData
    });
  } catch (error) {
    console.error('Snippet analysis error:', error);
    // Clean up the file in case of error
    if (req.file && fs.existsSync(req.file.path)) {
      try {
        await fs.promises.unlink(req.file.path);
        console.log('Temporary file cleaned up after error:', req.file.path);
      } catch (cleanupError) {
        console.error('Error cleaning up temporary file after error:', cleanupError);
      }
    }
    res.status(500).json({
      success: false,
      error: 'Failed to analyze audio snippet',
      details: error.message
    });
  }
});

// Add a simple test endpoint
app.get('/api/test', (req, res) => {
  console.log('Test endpoint hit');
  res.json({ 
    success: true,
    data: {
      message: 'API is working correctly',
      timestamp: new Date().toISOString()
    }
  });
});

// Add a simple file upload test endpoint
app.post('/api/test-upload', upload.single('file'), (req, res) => {
  console.log('Test upload endpoint hit');
  
  if (!req.file) {
    return res.status(400).json({ 
      success: false,
      error: 'No file provided' 
    });
  }
  
  console.log('Received file:', req.file);
  
  res.json({
    success: true,
    data: {
      message: 'File upload successful',
      file: {
        originalname: req.file.originalname,
        mimetype: req.file.mimetype,
        size: req.file.size,
        path: req.file.path
      }
    }
  });
});

// Add a default route for /api
app.get('/api', (req, res) => {
  // Check if the request accepts HTML
  if (req.accepts('html')) {
    res.send(`
      <!DOCTYPE html>
      <html>
        <head>
          <title>EliteSpeak API</title>
          <style>
            body {
              font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
              line-height: 1.6;
              color: #333;
              max-width: 800px;
              margin: 0 auto;
              padding: 20px;
            }
            h1 { color: #2196F3; }
            .endpoint {
              background: #f5f5f5;
              padding: 15px;
              border-radius: 5px;
              margin: 10px 0;
            }
            .method {
              font-weight: bold;
              color: #e91e63;
            }
          </style>
        </head>
        <body>
          <h1>EliteSpeak API</h1>
          <p>Welcome to the EliteSpeak API. Below are the available endpoints:</p>
          
          <div class="endpoint">
            <p><span class="method">POST</span> /api/analyze</p>
            <p>Upload and analyze audio files for speech analysis.</p>
          </div>
          
          <div class="endpoint">
            <p><span class="method">POST</span> /api/speech/analyze-snippet</p>
            <p>Analyze specific snippets of speech audio.</p>
          </div>
          
          <div class="endpoint">
            <p><span class="method">GET</span> /api/test</p>
            <p>Test endpoint to verify API connectivity.</p>
          </div>
          
          <div class="endpoint">
            <p><span class="method">POST</span> /api/test-upload</p>
            <p>Test endpoint for file upload functionality.</p>
          </div>
        </body>
      </html>
    `);
  } else {
    // Return JSON if HTML is not accepted
    res.json({
      name: 'EliteSpeak API',
      version: '1.0.0',
      endpoints: {
        '/api/analyze': {
          method: 'POST',
          description: 'Upload and analyze audio files for speech analysis'
        },
        '/api/speech/analyze-snippet': {
          method: 'POST',
          description: 'Analyze specific snippets of speech audio'
        },
        '/api/test': {
          method: 'GET',
          description: 'Test endpoint to verify API connectivity'
        },
        '/api/test-upload': {
          method: 'POST',
          description: 'Test endpoint for file upload functionality'
        }
      }
    });
  }
});

// Add 404 handler for API routes
app.use('/api/*', (req, res) => {
  res.status(404).json({
    error: 'Not Found',
    message: `Cannot ${req.method} ${req.url}`,
    availableEndpoints: [
      '/api/analyze',
      '/api/speech/analyze-snippet',
      '/api/test',
      '/api/test-upload'
    ]
  });
});

// Add general 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not Found',
    message: `Cannot ${req.method} ${req.url}`
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Server error:', err);
  
  // Handle Multer errors specifically
  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(413).json({
        success: false,
        error: 'File too large',
        details: `File size exceeds the ${50 / (1024 * 1024)}MB limit`
      });
    }
    return res.status(400).json({
      success: false,
      error: 'File upload error',
      details: err.message
    });
  }
  
  // Handle other errors
  res.status(500).json({
    success: false,
    error: 'Server error',
    details: err.message
  });
});

// Start server with improved logging
app.listen(port, () => {
  console.log('\n==================================');
  console.log(`🚀 Server running on port ${port}`);
  console.log(`📁 Uploads directory: ${uploadsDir}`);
  console.log(`🔗 API URL: http://localhost:${port}/api`);
  console.log('==================================\n');
}); 
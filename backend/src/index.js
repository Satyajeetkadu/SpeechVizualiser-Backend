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
      throw new Error('No audio file path provided - pitch analysis requires an audio file');
    }

    // Check if file exists
    if (!fs.existsSync(audioPath)) {
      throw new Error('Audio file not found - cannot perform pitch analysis');
    }

    // Convert audio to mono 16kHz WAV using ffmpeg
    processedAudioPath = path.join(
      path.dirname(audioPath),
      path.basename(audioPath, path.extname(audioPath)) + '_processed.wav'
    );

    console.log('Converting audio to WAV format...');
    try {
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
    } catch (ffmpegError) {
      console.error('Error converting audio:', ffmpegError);
      throw new Error('Failed to convert audio to WAV format: ' + ffmpegError.message);
    }

    // Check if processed file exists
    if (!fs.existsSync(processedAudioPath)) {
      throw new Error('Processed audio file was not created - FFmpeg conversion failed');
    }

    // Read the processed audio file
    let audioBuffer;
    try {
      audioBuffer = await fs.promises.readFile(processedAudioPath);
    } catch (readError) {
      console.error('Error reading processed audio file:', readError);
      throw new Error('Failed to read processed audio file: ' + readError.message);
    }
    
    // Convert Buffer to ArrayBuffer
    const arrayBuffer = audioBuffer.buffer.slice(
      audioBuffer.byteOffset,
      audioBuffer.byteOffset + audioBuffer.byteLength
    );

    try {
      console.log('Initializing audio context...');
      // Initialize Web Audio API context
      const audioContext = new AudioContext();
      
      console.log('Decoding audio data...');
      // Decode audio data
      const audioData = await audioContext.decodeAudioData(arrayBuffer);
      
      console.log('Configuring Meyda analyzer...');
      // Configure Meyda analyzer
      Meyda.bufferSize = 512;
      Meyda.sampleRate = audioData.sampleRate;
      
      // Get audio data as Float32Array
      const channelData = audioData.getChannelData(0);
      
      console.log('Processing audio in chunks...');
      // Process audio in chunks
      const frameSize = 512;
      const pitchData = [];
      
      // Process audio data in chunks
      for (let i = 0; i < channelData.length; i += frameSize) {
        if (i + frameSize > channelData.length) break;
        
        const frame = channelData.slice(i, i + frameSize);
        const features = Meyda.extract(['rms', 'zcr'], frame);
        
        // Simple pitch estimation based on zero-crossing rate
        // Higher ZCR generally indicates higher frequency
        pitchData.push({
          time: i / audioData.sampleRate,
          pitch: features.zcr * 100, // Scale for visualization
          energy: features.rms
        });
      }
      
      if (pitchData.length === 0) {
        throw new Error('No pitch data could be extracted from the audio');
      }
      
      console.log('Extracting audio snippet for playback...');
      // Extract audio snippet for playback
      const audioSnippetUrl = await extractAudioSnippet(audioPath);
      
      console.log('Processing pitch data...');
      // Process pitch data with transcription
      const processedData = processPitchData(pitchData);
      
      // Combine with transcription data
      const result = {
        sentences: processedData,
        audioSnippetUrl
      };
      
      console.log('Pitch analysis complete with real data');
      return result;
    } catch (error) {
      console.error('Error in audio processing:', error);
      
      // Check if error is related to ALSA/audio device
      const isAudioDeviceError = error.message && (
        error.message.includes('ALSA') || 
        error.message.includes('PCM') || 
        error.message.includes('DeviceNotAvailable') ||
        error.message.includes('device')
      );
      
      if (isAudioDeviceError) {
        console.error('ALSA/audio device error detected - server environment may not support audio processing');
        throw new Error('Server environment does not support audio processing: ' + error.message);
      } else {
        throw new Error('Failed to process audio data: ' + error.message);
      }
    }
  } catch (error) {
    console.error('Error in pitch analysis:', error);
    throw new Error('Pitch analysis failed: ' + error.message);
  } finally {
    // Clean up processed audio file
    if (processedAudioPath && fs.existsSync(processedAudioPath)) {
      try {
        fs.unlinkSync(processedAudioPath);
        console.log('Cleaned up processed audio file');
      } catch (error) {
        console.error('Error deleting processed audio file:', error);
      }
    }
  }
}

// Function to generate synthetic pitch data when audio processing fails
function generateSyntheticPitchData(transcriptionData) {
  console.log('Generating synthetic pitch data based on transcription');
  
  // Extract sentences from transcription
  const sentences = [];
  
  if (transcriptionData && transcriptionData.words && transcriptionData.words.length > 0) {
    // Use the actual words from transcription with their timing
    const words = transcriptionData.words;
    let currentSentence = {
      text: '',
      words: [],
      start: parseFloat(words[0].start) || 0,
      end: 0
    };
    
    // Group words into sentences
    for (let i = 0; i < words.length; i++) {
      const word = words[i];
      
      // Get the word text, checking both word.word (ElevenLabs format) and word.text properties
      const wordText = word.word || word.text || '';
      
      // Add word to current sentence
      currentSentence.text += wordText + ' ';
      
      // Add word with synthetic pitch
      currentSentence.words.push({
        text: wordText,
        start: parseFloat(word.start) || 0,
        end: parseFloat(word.end) || 0,
        pitch: 100 + Math.random() * 50 // Random pitch between 100-150
      });
      
      // Update sentence end time
      currentSentence.end = parseFloat(word.end) || 0;
      
      // Check if this is the end of a sentence
      const isEndOfSentence = 
        (wordText && (
          wordText.endsWith('.') || 
          wordText.endsWith('!') || 
          wordText.endsWith('?')
        )) ||
        i === words.length - 1 || // Last word
        currentSentence.words.length > 15; // Limit sentence length
      
      if (isEndOfSentence) {
        // Calculate average pitch and variation
        const pitches = currentSentence.words.map(w => w.pitch);
        const avgPitch = pitches.reduce((sum, p) => sum + p, 0) / pitches.length;
        const pitchVariation = Math.sqrt(
          pitches.reduce((sum, p) => sum + Math.pow(p - avgPitch, 2), 0) / pitches.length
        );
        
        // Find emphasis points (words with higher pitch)
        const sortedWords = [...currentSentence.words].sort((a, b) => b.pitch - a.pitch);
        const emphasis = sortedWords.slice(0, 3).map(w => ({
          word: w.text,
          emphasis: (w.pitch - avgPitch) / avgPitch
        }));
        
        // Add completed sentence
        sentences.push({
          text: currentSentence.text.trim(),
          start: currentSentence.start,
          end: currentSentence.end,
          words: currentSentence.words,
          averagePitch: avgPitch,
          pitchVariation,
          emphasis
        });
        
        // Start a new sentence if not at the end
        if (i < words.length - 1) {
          currentSentence = {
            text: '',
            words: [],
            start: parseFloat(words[i+1].start) || 0,
            end: 0
          };
        }
      }
    }
  } else if (transcriptionData && transcriptionData.text) {
    // Fallback to using just the text
    const sentenceTexts = transcriptionData.text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    
    // Generate synthetic data for each sentence
    sentenceTexts.slice(0, 10).forEach((text, index) => {
      // Create words array with estimated timing
      const words = text.split(' ').map((word, wordIndex) => {
        return {
          text: word,
          start: index * 5 + wordIndex * 0.3,
          end: index * 5 + wordIndex * 0.3 + 0.25,
          pitch: 100 + Math.random() * 50
        };
      });
      
      sentences.push({
        text: text.trim(),
        start: index * 5,
        end: index * 5 + words.length * 0.3,
        words,
        averagePitch: 125,
        pitchVariation: 20,
        emphasis: words.map(w => ({ word: w.text, emphasis: Math.random() })).slice(0, 3)
      });
    });
  }
  
  console.log(`Generated ${sentences.length} synthetic sentences`);
  
  return {
    sentences: sentences.slice(0, 10), // Limit to 10 sentences
    audioSnippetUrl: null
  };
}

// Function to process raw pitch data into sentence-based format
function processPitchData(pitchData) {
  console.log('Processing pitch data with', pitchData.length, 'data points');
  
  // If no pitch data, return empty array
  if (!pitchData || pitchData.length === 0) {
    return [];
  }
  
  // Create synthetic sentences based on pitch patterns
  const sentences = [];
  const sentenceLength = Math.floor(pitchData.length / 10); // Divide data into ~10 sentences
  
  for (let i = 0; i < pitchData.length; i += sentenceLength) {
    if (i + 5 >= pitchData.length) break; // Ensure we have at least 5 points
    
    const sentencePitchData = pitchData.slice(i, i + sentenceLength);
    const startTime = sentencePitchData[0].time;
    const endTime = sentencePitchData[sentencePitchData.length - 1].time;
    
    // Generate synthetic words based on pitch data
    const words = [];
    const wordCount = Math.min(10, Math.floor(sentencePitchData.length / 5));
    const wordLength = sentencePitchData.length / wordCount;
    
    for (let j = 0; j < wordCount; j++) {
      const wordStart = j * wordLength;
      const wordEnd = (j + 1) * wordLength - 1;
      
      if (wordStart >= sentencePitchData.length) break;
      
      const wordPitchData = sentencePitchData.slice(wordStart, wordEnd + 1);
      if (wordPitchData.length === 0) continue;
      
      const wordStartTime = wordPitchData[0].time;
      const wordEndTime = wordPitchData[wordPitchData.length - 1].time;
      const avgPitch = wordPitchData.reduce((sum, p) => sum + p.pitch, 0) / wordPitchData.length;
      
      words.push({
        text: `word${j}`,
        start: wordStartTime,
        end: wordEndTime,
        pitch: avgPitch
      });
    }
    
    // Calculate average pitch and variation
    const pitches = sentencePitchData.map(p => p.pitch);
    const avgPitch = pitches.reduce((sum, p) => sum + p, 0) / pitches.length;
    const pitchVariation = Math.sqrt(
      pitches.reduce((sum, p) => sum + Math.pow(p - avgPitch, 2), 0) / pitches.length
    );
    
    // Find emphasis points (peaks in pitch)
    const emphasis = [];
    for (let j = 1; j < sentencePitchData.length - 1; j++) {
      if (sentencePitchData[j].pitch > sentencePitchData[j-1].pitch + 10 &&
          sentencePitchData[j].pitch > sentencePitchData[j+1].pitch + 10) {
        emphasis.push({
          word: `word${Math.floor(j / wordLength)}`,
          emphasis: (sentencePitchData[j].pitch - avgPitch) / avgPitch
        });
      }
    }
    
    sentences.push({
      text: `Sentence ${i / sentenceLength + 1}`,
      start: startTime,
      end: endTime,
      words,
      averagePitch: avgPitch,
      pitchVariation,
      emphasis: emphasis.slice(0, 3) // Keep only top 3 emphasis points
    });
  }
  
  return sentences;
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
    // First, check if the pitch data is valid
    if (!pitchData) {
      console.log("No pitch data provided for vocal archetypes analysis");
      return getDefaultVocalArchetypes();
    }
    
    console.log("Pitch data type:", typeof pitchData);
    console.log("Is pitch data array:", Array.isArray(pitchData));
    
    // Check pitch data structure for debugging
    if (pitchData && typeof pitchData === 'object') {
      const keys = Object.keys(pitchData);
      console.log("Pitch data keys:", keys.join(", "));
      
      // Check for nested structures
      if (pitchData.sentences) {
        console.log("Found sentences in pitch data, length:", pitchData.sentences.length);
      }
    }
    
    // Process raw pitch data to format expected by extractPitchFeatures
    let processedPitchData;
    
    // If pitchData contains raw pitch points (time + pitch)
    if (Array.isArray(pitchData) && pitchData.length > 0 && 'time' in pitchData[0] && 'pitch' in pitchData[0]) {
      console.log("Processing raw pitch points data");
      
      // Create segments with pitchValues for extractPitchFeatures
      const segments = [{
        text: transcript || "",
        duration: (pitchData[pitchData.length-1]?.time || 0) - (pitchData[0]?.time || 0),
        pitchValues: pitchData.map(p => p.pitch)
      }];
      
      processedPitchData = segments;
    }
    // If pitchData is already the processed pitch analysis with sentences 
    else if (pitchData && pitchData.sentences && Array.isArray(pitchData.sentences)) {
      console.log("Using sentences from pitch analysis");
      
      // Convert each sentence to a segment with pitchValues
      processedPitchData = pitchData.sentences.map(sentence => {
        // Get all pitch values from words in the sentence
        const pitchValues = sentence.words
          ? sentence.words
              .filter(word => word && typeof word.pitch === 'number')
              .map(word => word.pitch) 
          : [];
          
        return {
          text: sentence.text || "",
          duration: (sentence.end || 0) - (sentence.start || 0),
          pitchValues: pitchValues
        };
      });
    }
    // If pitchData is already in the expected format with pitchValues
    else if (Array.isArray(pitchData) && pitchData.some(item => item && item.pitchValues)) {
      console.log("Using existing processed pitch data with pitchValues");
      processedPitchData = pitchData;
    }
    // As a last resort, try to extract features directly
    else {
      console.log("Using custom feature extraction for pitch data");
      
      // Skip feature extraction and use default values
      const features = {
        averagePitch: 120,
        pitchVariability: 20,
        speakingRate: 150,
        pauseFrequency: 10
      };
      
      // Use GPT to analyze the vocal archetype based on transcript and features
      const prompt = `
You are an expert speech analyst who specializes in identifying vocal archetypes in speech. 
I need you to analyze the following transcript and classify the speaker 
into these three specific archetypes: The Valiant, The Caregiver, and The Sage.

Transcript: "${transcript}"

Archetypes:
1. The Valiant (Motivational, Energetic, Action-oriented)
   - Characterizes speakers who are motivational, energizing, and action-oriented
   - Uses uplifting language and dynamic delivery
   - Focuses on growth, achievement, and overcoming challenges
   - Emphasizes action steps and forward momentum

2. The Caregiver (Nurturing, Supportive, Relationship-focused)
   - Characterizes speakers who are nurturing, supportive, and relationship-focused
   - Uses empathetic language and warm delivery
   - Focuses on connection, wellbeing, and meeting needs
   - Emphasizes shared experiences and community

3. The Sage (Thoughtful, Measured, Knowledge-focused)
   - Characterizes speakers who are contemplative, measured, and knowledge-focused
   - Uses precise language and measured delivery
   - Focuses on insight, wisdom, and understanding
   - Emphasizes clarity, depth, and perspective

Based solely on the transcript content and style (not considering audio features), analyze the speaker's vocal archetype.

For your response, provide:
1. A percentage breakdown across all three archetypes (must total 100%)
2. The dominant archetype (the one with the highest percentage)
3. A brief analysis explaining why this archetype is dominant and how the other archetypes are or aren't present

Output the results in the following JSON format:
{
  "archetypes": [
    {"name": "The Valiant", "score": number, "color": "#FFC107"},
    {"name": "The Caregiver", "score": number, "color": "#E91E63"},
    {"name": "The Sage", "score": number, "color": "#2196F3"}
  ],
  "dominantArchetype": "The [Dominant Archetype Name]",
  "analysis": "Brief analysis explaining the results."
}`;

      try {
        const response = await openai.chat.completions.create({
          model: "gpt-4",
          messages: [
            { role: "system", content: prompt },
            { role: "user", content: transcript }
          ],
          temperature: 0.7,
          max_tokens: 1500
        });
        
        const result = JSON.parse(response.choices[0].message.content);
        console.log("Vocal archetype analysis response:", result);
        return result;
      } catch (error) {
        console.error("Error in vocal archetype GPT analysis:", error);
        return getDefaultVocalArchetypes();
      }
    }
    
    // Now extract features using the properly formatted data
    try {
      console.log("Extracting features from processed pitch data");
      const features = extractPitchFeatures(processedPitchData);
      
      // Use GPT to analyze the vocal archetype based on transcript and features
      const prompt = `
You are an expert speech analyst who specializes in identifying vocal archetypes in speech. 
I need you to analyze the following transcript and speech features and classify the speaker 
into these three specific archetypes: The Valiant, The Caregiver, and The Sage.

Transcript: "${transcript}"

Speech Features:
- Average Pitch: ${features.averagePitch} (Higher values indicate higher pitch)
- Pitch Variability: ${features.pitchVariability} (Higher values indicate more varied intonation)
- Speaking Rate: ${features.speakingRate} words per minute (Average is 120-150)
- Pause Frequency: ${features.pauseFrequency} pauses per minute (Higher values indicate more frequent pauses)

Archetypes:
1. The Valiant (Motivational, Energetic, Action-oriented)
   - Characterizes speakers who are motivational, energizing, and action-oriented
   - Typically has higher pitch, greater variability, faster speaking rate
   - Uses uplifting language and dynamic delivery
   - Focuses on growth, achievement, and overcoming challenges
   - Emphasizes action steps and forward momentum

2. The Caregiver (Nurturing, Supportive, Relationship-focused)
   - Characterizes speakers who are nurturing, supportive, and relationship-focused
   - Typically has a warm, steady pitch with moderate variability
   - Uses empathetic language and warm delivery
   - Focuses on connection, wellbeing, and meeting needs
   - Emphasizes shared experiences and community

3. The Sage (Thoughtful, Measured, Knowledge-focused)
   - Characterizes speakers who are contemplative, measured, and knowledge-focused
   - Typically has a lower, more consistent pitch with strategic pauses
   - Uses precise language and measured delivery
   - Focuses on insight, wisdom, and understanding
   - Emphasizes clarity, depth, and perspective

Based on both the transcript content and the speech features, analyze the speaker's vocal archetype.

For your response, provide:
1. A percentage breakdown across all three archetypes (must total 100%)
2. The dominant archetype (the one with the highest percentage)
3. A brief analysis explaining why this archetype is dominant and how the other archetypes are or aren't present

Output the results in the following JSON format:
{
  "archetypes": [
    {"name": "The Valiant", "score": number, "color": "#FFC107"},
    {"name": "The Caregiver", "score": number, "color": "#E91E63"},
    {"name": "The Sage", "score": number, "color": "#2196F3"}
  ],
  "dominantArchetype": "The [Dominant Archetype Name]",
  "analysis": "Brief analysis explaining the results."
}`;

      const response = await openai.chat.completions.create({
        model: "gpt-4",
        messages: [
          { role: "system", content: prompt },
          { role: "user", content: transcript }
        ],
        temperature: 0.7,
        max_tokens: 1500
      });
      
      const result = JSON.parse(response.choices[0].message.content);
      console.log("Vocal archetype analysis response:", result);
      return result;
    } catch (error) {
      console.error("Error extracting pitch features:", error);
      // Fall back to text-only analysis
      return getDefaultVocalArchetypes();
    }
  } catch (error) {
    console.error("Error in vocal archetypes analysis:", error);
    return getDefaultVocalArchetypes();
  }
}

// Helper function to extract speech features from pitch data
function extractPitchFeatures(pitchData) {
  try {
    console.log("Extracting pitch features from data...");
    
    // Check if pitchData is an array of sentences (synthetic data format)
    if (Array.isArray(pitchData) && pitchData.length > 0 && pitchData[0].text) {
      console.log("Processing synthetic pitch data format (array of sentences)");
      
      // Calculate average pitch from words in sentences
      let allPitchValues = [];
      let totalWords = 0;
      let totalDuration = 0;
      let pauseCount = 0;
      
      // Extract pitch values from all words in all sentences
      pitchData.forEach((sentence, idx) => {
        if (sentence.words && Array.isArray(sentence.words)) {
          // Add pitch values from words
          const sentencePitches = sentence.words
            .filter(word => word && typeof word.pitch === 'number')
            .map(word => word.pitch);
          
          allPitchValues = allPitchValues.concat(sentencePitches);
          totalWords += sentence.words.length;
          
          // Calculate duration
          if (typeof sentence.start === 'number' && typeof sentence.end === 'number') {
            totalDuration += (sentence.end - sentence.start);
          }
          
          // Count pauses between sentences
          if (idx > 0 && pitchData[idx-1] && 
              typeof sentence.start === 'number' && 
              typeof pitchData[idx-1].end === 'number') {
            const pauseDuration = sentence.start - pitchData[idx-1].end;
            if (pauseDuration > 0.2) {
              pauseCount++;
            }
          }
        }
      });
      
      // Calculate average pitch
      const averagePitch = allPitchValues.length > 0 
        ? Math.round(allPitchValues.reduce((sum, val) => sum + val, 0) / allPitchValues.length)
        : 0;
      
      // Calculate pitch variability (standard deviation)
      let pitchVariability = 0;
      if (allPitchValues.length > 0) {
        const squaredDiffs = allPitchValues.map(val => Math.pow(val - averagePitch, 2));
        const variance = squaredDiffs.reduce((sum, val) => sum + val, 0) / allPitchValues.length;
        pitchVariability = Math.round(Math.sqrt(variance));
      }
      
      // Calculate speaking rate (words per minute)
      const speakingRate = totalDuration > 0 
        ? Math.round((totalWords / totalDuration) * 60) 
        : 0;
      
      // Calculate pause frequency
      const pauseFrequency = totalDuration > 0 
        ? Math.round((pauseCount / totalDuration) * 60) 
        : 0;
      
      console.log(`Extracted features from synthetic data: avgPitch=${averagePitch}, variability=${pitchVariability}, rate=${speakingRate}, pauses=${pauseFrequency}`);
      
      return {
        averagePitch,
        pitchVariability,
        speakingRate,
        pauseFrequency
      };
    }
    // Check if pitchData has sentences property (real data format from analyzePitch)
    else if (pitchData && pitchData.sentences && Array.isArray(pitchData.sentences)) {
      console.log("Processing real pitch data format (object with sentences property)");
      return extractPitchFeatures(pitchData.sentences);
    }
    // Check if pitchData has pitchValues property (original format expected by the function)
    else if (pitchData && Array.isArray(pitchData) && pitchData.some(segment => segment.pitchValues)) {
      console.log("Processing original pitch data format with pitchValues");
      
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
    }
    // Fallback for unknown format
    else {
      console.log("Unknown pitch data format, returning default values");
      console.log("Pitch data type:", typeof pitchData);
      console.log("Is array:", Array.isArray(pitchData));
      if (pitchData) {
        console.log("First item properties:", Object.keys(pitchData[0] || {}).join(", "));
      }
      
      return {
        averagePitch: 120,
        pitchVariability: 20,
        speakingRate: 150,
        pauseFrequency: 10
      };
    }
  } catch (error) {
    console.error("Error extracting pitch features:", error);
    return {
      averagePitch: 120,
      pitchVariability: 20,
      speakingRate: 150,
      pauseFrequency: 10
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
        
        return phrase;
      });
    } catch (e) {
      console.error('Failed to parse AI analysis', e);
      parsedAnalysis = { preciseLanguage: [] };
    }
    
    // Process highlighted phrases
    let highlightedPhrases = [];
    
    try {
      // Extract raw precise language instances from AI response
      console.log('\n=== Raw Precise Phrases ===');
      console.log(JSON.stringify(parsedAnalysis.preciseLanguage || [], null, 2));
      
      // For each precise phrase, find its containing sentence
      const sentences = transcriptionText.split(/[.!?]+/).filter(s => s.trim().length > 0);
      
      // Process each phrase
      for (const phrase of (parsedAnalysis.preciseLanguage || [])) {
        console.log(`\nPhrase: "${phrase.phrase}"`);
        
        // Find which sentence contains this phrase
        const phraseText = phrase.phrase;
        let foundSentence = null;
        let sentenceText = null;
        
        for (const sentence of sentences) {
          if (sentence.includes(phraseText)) {
            foundSentence = sentence.trim();
            sentenceText = foundSentence;
            break;
          }
        }
        
        if (!sentenceText) {
          console.log(`Could not find sentence containing phrase "${phraseText}"`);
          continue;
        }
        
        console.log(`In sentence: "${sentenceText}"`);
        console.log(`Original indices: ${phrase.startIndex}-${phrase.endIndex}`);
        
        // Calculate relative indices within the sentence
        const sentenceStartIndex = transcriptionText.indexOf(sentenceText);
        const relativeStartIndex = phrase.startIndex - sentenceStartIndex;
        const relativeEndIndex = phrase.endIndex - sentenceStartIndex;
        
        console.log(`Relative indices: ${relativeStartIndex}-${relativeEndIndex}`);
        
        // Extract the text at these indices to verify
        const extractedText = sentenceText.substring(relativeStartIndex, relativeEndIndex);
        console.log(`Extracted text: "${extractedText}"`);
        
        // If the extracted text doesn't match the phrase, try to find the exact phrase in the sentence
        if (extractedText !== phraseText) {
          console.log(`Text mismatch! Searching for exact phrase in sentence...`);
          
          // Try to find the exact phrase in the sentence
          const phraseIndexInSentence = sentenceText.indexOf(phraseText);
          if (phraseIndexInSentence !== -1) {
            console.log(`Found exact phrase at indices ${phraseIndexInSentence}-${phraseIndexInSentence + phraseText.length}`);
            console.log(`New extracted text: "${sentenceText.substring(phraseIndexInSentence, phraseIndexInSentence + phraseText.length)}"`);
            
            // Update the highlight data
            const highlight = {
              text: phraseText,
              start: phraseIndexInSentence,
              end: phraseIndexInSentence + phraseText.length,
              significance: phrase.significance,
              category: phrase.category
            };
            
            console.log(`Highlight data being sent to frontend:`, highlight);
            
            // Add to the list of highlighted phrases for this sentence
            let existingInstance = highlightedPhrases.find(p => p.sentence === sentenceText);
            if (existingInstance) {
              existingInstance.highlights.push(highlight);
            } else {
              highlightedPhrases.push({
                sentence: sentenceText,
                highlights: [highlight]
              });
            }
          } else {
            console.log(`Could not find exact phrase in sentence`);
          }
        } else {
          // If the text matched, use the original indices
          const highlight = {
            text: phraseText,
            start: relativeStartIndex,
            end: relativeEndIndex,
            significance: phrase.significance,
            category: phrase.category
          };
          
          console.log(`Highlight data being sent to frontend:`, highlight);
          
          // Add to the list of highlighted phrases for this sentence
          let existingInstance = highlightedPhrases.find(p => p.sentence === sentenceText);
          if (existingInstance) {
            existingInstance.highlights.push(highlight);
          } else {
            highlightedPhrases.push({
              sentence: sentenceText,
              highlights: [highlight]
            });
          }
        }
      }
      
      console.log('\n=== Final Processed Instances ===');
      console.log(JSON.stringify(highlightedPhrases, null, 2));
    } catch (highlightError) {
      console.error('Error processing highlighted phrases:', highlightError);
      highlightedPhrases = [];
    }
    
    // Analyze pronoun usage
    const pronounAnalysis = analyzePronounUsage(transcriptionText);
    
    // Analyze erosion tags
    const erosionTagsResult = await analyzeErosionTags(transcriptionText);
    console.log('\n=== Erosion Tags Analysis Complete ===');
    
    // Calculate final language precision score (weighted average)
    const baseScore = 70; // Base score
    const preciseLanguageBonus = Math.min(20, (parsedAnalysis.preciseLanguage?.length || 0) * 5);
    const pronounPenalty = Math.max(0, Math.min(20, pronounAnalysis.anxiousnessScore / 5));
    const erosionTagsPenalty = Math.max(0, 20 - (erosionTagsResult?.score || 0) / 5);
    
    const finalScore = Math.min(100, Math.max(0, 
      baseScore + preciseLanguageBonus - pronounPenalty - erosionTagsPenalty
    ));
    
    console.log('\n=== Final Language Precision Score ===');
    console.log('Score:', Math.round(finalScore));
    
    // Analyze pitch patterns
    let pitchAnalysis = null;
    try {
      console.log('\n=== Starting Pitch Analysis ===');
      
      // Check if audio path is available
      const audioPath = transcriptionData.audioPath;
      if (!audioPath) {
        throw new Error('No audio path provided for pitch analysis');
      }
      
      // Attempt to analyze pitch patterns
      pitchAnalysis = await analyzePitch(audioPath, transcriptionData);
    } catch (pitchError) {
      console.error('Error processing audio:', pitchError);
      pitchAnalysis = { 
        error: pitchError.message || 'Unknown error in pitch analysis',
        errorType: 'PitchAnalysisError',
        errorDetail: pitchError.stack
      };
    }
    
    console.log('\n=== Analysis Complete ===');
    
    // Check if pitchAnalysis exists and isn't an error
    const hasPitchData = pitchAnalysis && !pitchAnalysis.error;
    
    // Analyze vocal archetypes (if pitch data is available)
    let vocalArchetypesAnalysis;
    try {
      console.log('Analyzing vocal archetypes...');
      vocalArchetypesAnalysis = await analyzeVocalArchetypes(transcriptionText, hasPitchData ? pitchAnalysis : null);
      console.log('Vocal archetype analysis response:', vocalArchetypesAnalysis);
    } catch (vocalError) {
      console.error('Error in vocal archetypes analysis:', vocalError);
      vocalArchetypesAnalysis = getDefaultVocalArchetypes();
    }
    
    // Analyze visual language
    let visualLanguageAnalysis;
    try {
      visualLanguageAnalysis = await analyzeVisualLanguage(transcriptionText);
      console.log('Visual Language Analysis complete.');
    } catch (visualError) {
      console.error('Error in visual language analysis:', visualError);
      visualLanguageAnalysis = {
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
    
    // Analyze silence comfort (if pitch data is available)
    let silenceComfortAnalysis;
    try {
      silenceComfortAnalysis = {
        // Process in real-time
      };
    } catch (silenceError) {
      console.error('Error in silence comfort analysis:', silenceError);
      silenceComfortAnalysis = {
        groups: [],
        overallScore: 75
      };
    }
    
    // Compile and return final analysis results
    return {
      languagePrecision: {
        instances: highlightedPhrases,
        pronouns: pronounAnalysis,
        erosionTags: erosionTagsResult,
        score: Math.round(finalScore)
      },
      pitch: pitchAnalysis,
      vocalArchetypes: vocalArchetypesAnalysis,
      visualLanguage: visualLanguageAnalysis,
      silenceComfort: (hasPitchData && pitchAnalysis.sentences) ? processSilenceComfort(transcriptionData) : {
        groups: [],
        overallScore: 75
      },
      thoughtFlow: analyzeThoughtFlow(transcriptionText)
    };
  } catch (error) {
    console.error('Error analyzing transcript:', error);
    return {
      languagePrecision: {
        instances: [],
        pronouns: {
          counts: { i: 0, we: 0, you: 0, other: 0 },
          distribution: { self: 33, inclusive: 33, other: 34 },
          examples: { self: [], inclusive: [], other: [] },
          anxiousnessScore: 70
        },
        erosionTags: {
          erosionTags: []
        },
        score: 70
      },
      pitch: { 
        error: error.message || 'Unknown error in analysis',
        errorType: 'AnalysisError',
        errorDetail: error.stack
      },
      vocalArchetypes: getDefaultVocalArchetypes(),
      visualLanguage: {
        breakdown: {
          visual: 40,
          functional: 60,
          examples: {
            visual: [],
            functional: []
          }
        },
        score: 65
      },
      silenceComfort: {
        groups: [],
        overallScore: 75
      },
      thoughtFlow: {
        structure: [
          { progress: 0, relevance: 1, label: "Opening" },
          { progress: 50, relevance: 2, label: "Main Content" },
          { progress: 100, relevance: 1, label: "Closing" }
        ],
        coherenceScore: 75
      }
    };
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
      
      // Add the audio path to the transcription data for pitch analysis
      transcriptionData.audioPath = req.file.path;
      
      // Analyze the transcript (this will also handle pitch analysis internally)
      const analysisData = await analyzeTranscript(transcriptionData);
      
      // Build the response with all analysis data
      const responseData = {
        transcription: transcriptionData,
        analysis: {
          languagePrecision: analysisData.languagePrecision,
          pitch: analysisData.pitch.error ? [] : analysisData.pitch.sentences || [],
          audioSnippetUrl: analysisData.pitch.error ? null : analysisData.pitch.audioSnippetUrl,
          silenceComfort: analysisData.silenceComfort,
          erosionTags: analysisData.languagePrecision.erosionTags,
          vocalArchetypes: analysisData.vocalArchetypes,
          visualLanguage: analysisData.visualLanguage,
          thoughtFlow: analysisData.thoughtFlow,
          pitchError: analysisData.pitch.error || null
        }
      };
      
      console.log(`\n=== Response Data Structure ===`);
      console.log(`- transcription: Present`);
      console.log(`- analysis.languagePrecision: Present`);
      console.log(`- analysis.pitch: ${analysisData.pitch.error ? 'Error: ' + analysisData.pitch.error : 'Array with ' + (responseData.analysis.pitch?.length || 0) + ' items'}`);
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
    transcription.audioPath = req.file.path;
    transcription.isSnippet = true;
    transcription.startTime = parseFloat(req.body.startTime) || 0;
    transcription.endTime = parseFloat(req.body.endTime) || 0;

    // Analyze the transcription (this will also handle pitch analysis internally)
    const analysis = await analyzeTranscript(transcription);
    
    console.log(`\n=== Snippet Analysis Results Structure ===`);
    console.log(`Pitch Data Structure:`);
    console.log(`- pitch: ${analysis.pitch.error ? 'Error: ' + analysis.pitch.error : 'Array with ' + (analysis.pitch.sentences?.length || 0) + ' items'}`);
    console.log(`- audioSnippetUrl: ${analysis.pitch.error ? 'Not available due to error' : analysis.pitch.audioSnippetUrl || 'Not available'}`);
    
    // Build the complete response with all analysis data
    const responseData = {
      transcription,
      analysis: {
        languagePrecision: analysis.languagePrecision,
        pitch: analysis.pitch.error ? [] : analysis.pitch.sentences || [],
        audioSnippetUrl: analysis.pitch.error ? null : analysis.pitch.audioSnippetUrl,
        silenceComfort: analysis.silenceComfort,
        erosionTags: analysis.languagePrecision.erosionTags,
        vocalArchetypes: analysis.vocalArchetypes,
        visualLanguage: analysis.visualLanguage,
        thoughtFlow: analysis.thoughtFlow,
        pitchError: analysis.pitch.error || null
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
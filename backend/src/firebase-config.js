import admin from 'firebase-admin';
import dotenv from 'dotenv';

dotenv.config();

// Initialize Firebase Admin with service account
// You'll need to create a service account and download the JSON file from Firebase console
try {
  // Check if app is already initialized to prevent multiple initializations
  if (!admin.apps.length) {
    // If you have a service account JSON file, use this:
    // const serviceAccount = require('../path-to-your-service-account.json');
    // admin.initializeApp({
    //   credential: admin.credential.cert(serviceAccount)
    // });

    // For deployment, it's better to use environment variables:
    admin.initializeApp({
      credential: admin.credential.cert({
        projectId: process.env.FIREBASE_PROJECT_ID,
        clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
        privateKey: process.env.FIREBASE_PRIVATE_KEY?.replace(/\\n/g, '\n')
      })
    });
    
    console.log('Firebase Admin initialized successfully');
  }
} catch (error) {
  console.error('Error initializing Firebase Admin:', error);
}

const db = admin.firestore();
const auth = admin.auth();

export { admin, db, auth }; 
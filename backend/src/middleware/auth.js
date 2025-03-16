import { auth } from '../firebase-config.js';

/**
 * Middleware to verify Firebase authentication token
 */
export const verifyToken = async (req, res, next) => {
  // Get the authorization header
  const authHeader = req.headers.authorization;
  
  // Check if public route (optional authentication)
  const isPublicRoute = req.path === '/api/test' || req.path === '/api/health';
  
  if (!authHeader && !isPublicRoute) {
    return res.status(401).json({ error: 'Unauthorized', details: 'No authorization token provided' });
  }
  
  if (!authHeader && isPublicRoute) {
    // For public routes, continue without authentication
    req.user = null;
    return next();
  }
  
  try {
    // Format of the authorization header should be "Bearer <token>"
    const token = authHeader.split(' ')[1];
    
    if (!token) {
      return res.status(401).json({ error: 'Unauthorized', details: 'Invalid token format' });
    }
    
    // Verify the token with Firebase Auth
    const decodedToken = await auth.verifyIdToken(token);
    
    // Add the user information to the request object
    req.user = {
      uid: decodedToken.uid,
      email: decodedToken.email,
      name: decodedToken.name
    };
    
    next();
  } catch (error) {
    console.error('Error verifying token:', error);
    
    if (isPublicRoute) {
      // For public routes, continue without authentication if token is invalid
      req.user = null;
      return next();
    }
    
    return res.status(401).json({ 
      error: 'Unauthorized', 
      details: 'Invalid or expired token' 
    });
  }
}; 
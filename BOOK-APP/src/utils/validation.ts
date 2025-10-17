/**
 * Validation Utilities
 * Provides comprehensive validation functions for forms, data, and user input
 * @module utils/validation
 */

import { z } from 'zod';

// ==================== EMAIL VALIDATION ====================

/**
 * Validates email format using regex
 * @param email - Email address to validate
 * @returns true if valid email format
 */
export const isValidEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

// ==================== PASSWORD VALIDATION ====================

/**
 * Password validation requirements
 */
export interface PasswordRequirements {
  minLength: boolean;
  hasUppercase: boolean;
  hasLowercase: boolean;
  hasNumber: boolean;
  hasSpecialChar: boolean;
}

/**
 * Validates password strength and returns detailed requirements
 * @param password - Password to validate
 * @returns Object with validation results for each requirement
 */
export const validatePassword = (password: string): PasswordRequirements => {
  return {
    minLength: password.length >= 8,
    hasUppercase: /[A-Z]/.test(password),
    hasLowercase: /[a-z]/.test(password),
    hasNumber: /[0-9]/.test(password),
    hasSpecialChar: /[!@#$%^&*(),.?":{}|<>]/.test(password),
  };
};

/**
 * Checks if password meets all requirements
 * @param password - Password to check
 * @returns true if password is strong enough
 */
export const isStrongPassword = (password: string): boolean => {
  const requirements = validatePassword(password);
  return Object.values(requirements).every((req) => req === true);
};

/**
 * Gets password strength score (0-5)
 * @param password - Password to score
 * @returns Strength score from 0 (weakest) to 5 (strongest)
 */
export const getPasswordStrength = (password: string): number => {
  const requirements = validatePassword(password);
  return Object.values(requirements).filter((req) => req === true).length;
};

// ==================== FILE VALIDATION ====================

/**
 * Validates file size
 * @param sizeInBytes - File size in bytes
 * @param maxSizeInMB - Maximum allowed size in MB (default: 200MB)
 * @returns true if file size is within limit
 */
export const isValidFileSize = (
  sizeInBytes: number,
  maxSizeInMB: number = 200
): boolean => {
  const maxSizeInBytes = maxSizeInMB * 1024 * 1024;
  return sizeInBytes <= maxSizeInBytes;
};

/**
 * Validates file MIME type
 * @param mimeType - File MIME type
 * @param allowedTypes - Array of allowed MIME types
 * @returns true if MIME type is allowed
 */
export const isValidMimeType = (
  mimeType: string,
  allowedTypes: string[]
): boolean => {
  return allowedTypes.includes(mimeType);
};

/**
 * Common MIME types for the application
 */
export const MIME_TYPES = {
  PDF: 'application/pdf',
  EPUB: 'application/epub+zip',
  MOBI: 'application/x-mobipocket-ebook',
  TXT: 'text/plain',
  IMAGE_JPEG: 'image/jpeg',
  IMAGE_PNG: 'image/png',
  IMAGE_WEBP: 'image/webp',
} as const;

/**
 * Validates PDF file
 * @param file - File object with mimeType and size
 * @param maxSizeInMB - Maximum size in MB
 * @returns Validation result with error message if invalid
 */
export const validatePDFFile = (
  file: { mimeType: string; size: number },
  maxSizeInMB: number = 200
): { valid: boolean; error?: string } => {
  if (!isValidMimeType(file.mimeType, [MIME_TYPES.PDF])) {
    return { valid: false, error: 'File must be a PDF' };
  }

  if (!isValidFileSize(file.size, maxSizeInMB)) {
    return {
      valid: false,
      error: `File size must be less than ${maxSizeInMB}MB`,
    };
  }

  return { valid: true };
};

// ==================== TEXT VALIDATION ====================

/**
 * Validates text length
 * @param text - Text to validate
 * @param minLength - Minimum length
 * @param maxLength - Maximum length
 * @returns true if text length is within range
 */
export const isValidLength = (
  text: string,
  minLength: number,
  maxLength: number
): boolean => {
  const length = text.trim().length;
  return length >= minLength && length <= maxLength;
};

/**
 * Validates annotation text
 * @param text - Annotation text
 * @returns Validation result
 */
export const validateAnnotationText = (text: string): {
  valid: boolean;
  error?: string;
} => {
  const trimmed = text.trim();

  if (trimmed.length === 0) {
    return { valid: false, error: 'Annotation text cannot be empty' };
  }

  if (trimmed.length > 5000) {
    return {
      valid: false,
      error: 'Annotation text must be less than 5000 characters',
    };
  }

  return { valid: true };
};

// ==================== URL VALIDATION ====================

/**
 * Validates URL format
 * @param url - URL to validate
 * @returns true if valid URL
 */
export const isValidURL = (url: string): boolean => {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
};

/**
 * Validates HTTP/HTTPS URL
 * @param url - URL to validate
 * @returns true if valid HTTP(S) URL
 */
export const isValidHttpURL = (url: string): boolean => {
  if (!isValidURL(url)) return false;
  const urlObj = new URL(url);
  return urlObj.protocol === 'http:' || urlObj.protocol === 'https:';
};

// ==================== ZOD SCHEMAS ====================

/**
 * Zod schema for login form
 */
export const loginSchema = z.object({
  email: z.string().email('Invalid email address'),
  password: z.string().min(1, 'Password is required'),
});

/**
 * Zod schema for signup form
 */
export const signupSchema = z
  .object({
    fullName: z.string().min(2, 'Name must be at least 2 characters'),
    email: z.string().email('Invalid email address'),
    password: z
      .string()
      .min(8, 'Password must be at least 8 characters')
      .regex(/[A-Z]/, 'Password must contain an uppercase letter')
      .regex(/[a-z]/, 'Password must contain a lowercase letter')
      .regex(/[0-9]/, 'Password must contain a number'),
    confirmPassword: z.string(),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: "Passwords don't match",
    path: ['confirmPassword'],
  });

/**
 * Zod schema for annotation
 */
export const annotationSchema = z.object({
  content: z
    .string()
    .min(1, 'Annotation cannot be empty')
    .max(5000, 'Annotation must be less than 5000 characters'),
  pageNumber: z.number().int().positive('Page number must be positive'),
  color: z.string().regex(/^#[0-9A-Fa-f]{6}$/, 'Invalid color format'),
});

/**
 * Zod schema for book metadata
 */
export const bookMetadataSchema = z.object({
  title: z.string().min(1, 'Title is required').max(500, 'Title too long'),
  author: z.string().max(300, 'Author name too long').optional(),
  description: z.string().max(5000, 'Description too long').optional(),
  totalPages: z
    .number()
    .int()
    .positive('Total pages must be positive')
    .optional(),
  isbn: z.string().regex(/^[\d-]{10,17}$/, 'Invalid ISBN format').optional(),
});

// ==================== SANITIZATION ====================

/**
 * Sanitizes text input by removing dangerous characters
 * @param text - Text to sanitize
 * @returns Sanitized text
 */
export const sanitizeText = (text: string): string => {
  return text
    .replace(/[<>]/g, '') // Remove angle brackets
    .replace(/javascript:/gi, '') // Remove javascript: protocol
    .trim();
};

/**
 * Sanitizes filename for storage
 * @param filename - Original filename
 * @returns Safe filename
 */
export const sanitizeFilename = (filename: string): string => {
  return filename
    .replace(/[^a-zA-Z0-9._-]/g, '_') // Replace special chars with underscore
    .replace(/_{2,}/g, '_') // Replace multiple underscores with single
    .toLowerCase();
};

// ==================== NUMBER VALIDATION ====================

/**
 * Validates page number
 * @param pageNumber - Page number to validate
 * @param totalPages - Total pages in document
 * @returns true if valid page number
 */
export const isValidPageNumber = (
  pageNumber: number,
  totalPages: number
): boolean => {
  return (
    Number.isInteger(pageNumber) &&
    pageNumber >= 1 &&
    pageNumber <= totalPages
  );
};

/**
 * Validates coordinates for annotations
 * @param coords - Coordinates object
 * @returns true if valid coordinates
 */
export const isValidCoordinates = (coords: {
  x: number;
  y: number;
  width: number;
  height: number;
}): boolean => {
  return (
    typeof coords.x === 'number' &&
    typeof coords.y === 'number' &&
    typeof coords.width === 'number' &&
    typeof coords.height === 'number' &&
    coords.x >= 0 &&
    coords.y >= 0 &&
    coords.width > 0 &&
    coords.height > 0 &&
    coords.x + coords.width <= 1 &&
    coords.y + coords.height <= 1
  );
};

// ==================== EXPORT ====================

export type { z };

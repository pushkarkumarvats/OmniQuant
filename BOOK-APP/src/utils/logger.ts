/**
 * Logger Utility
 * Centralized logging system with different levels and remote logging support
 * @module utils/logger
 */

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LogEntry {
  level: LogLevel;
  message: string;
  timestamp: Date;
  data?: any;
  stack?: string;
  userId?: string;
  context?: Record<string, any>;
}

/**
 * Logger class for structured logging
 */
class Logger {
  private isDevelopment: boolean;
  private logs: LogEntry[] = [];
  private maxLogs: number = 1000;
  private remoteLoggingEnabled: boolean = false;

  constructor() {
    this.isDevelopment = __DEV__;
  }

  /**
   * Logs a debug message (only in development)
   * @param message - Log message
   * @param data - Additional data
   */
  debug(message: string, data?: any): void {
    if (this.isDevelopment) {
      this.log('debug', message, data);
      console.log(`[DEBUG] ${message}`, data || '');
    }
  }

  /**
   * Logs an info message
   * @param message - Log message
   * @param data - Additional data
   */
  info(message: string, data?: any): void {
    this.log('info', message, data);
    console.log(`[INFO] ${message}`, data || '');
  }

  /**
   * Logs a warning message
   * @param message - Log message
   * @param data - Additional data
   */
  warn(message: string, data?: any): void {
    this.log('warn', message, data);
    console.warn(`[WARN] ${message}`, data || '');
  }

  /**
   * Logs an error message
   * @param message - Error message
   * @param error - Error object or data
   */
  error(message: string, error?: Error | any): void {
    const stack = error instanceof Error ? error.stack : undefined;
    this.log('error', message, error, stack);
    console.error(`[ERROR] ${message}`, error || '');

    // Send to remote logging service in production
    if (!this.isDevelopment && this.remoteLoggingEnabled) {
      this.sendToRemote('error', message, error, stack);
    }
  }

  /**
   * Internal logging method
   */
  private log(
    level: LogLevel,
    message: string,
    data?: any,
    stack?: string
  ): void {
    const entry: LogEntry = {
      level,
      message,
      timestamp: new Date(),
      data,
      stack,
    };

    this.logs.push(entry);

    // Maintain max log size
    if (this.logs.length > this.maxLogs) {
      this.logs.shift();
    }
  }

  /**
   * Sends log to remote service (Sentry, LogRocket, etc.)
   */
  private async sendToRemote(
    level: LogLevel,
    message: string,
    data?: any,
    stack?: string
  ): Promise<void> {
    // Implement remote logging here
    // Example: Send to Sentry, LogRocket, or custom backend
    try {
      // await fetch('https://your-logging-service.com/logs', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ level, message, data, stack }),
      // });
    } catch (err) {
      // Fail silently to avoid logging loops
    }
  }

  /**
   * Gets all stored logs
   */
  getLogs(level?: LogLevel): LogEntry[] {
    if (level) {
      return this.logs.filter((log) => log.level === level);
    }
    return this.logs;
  }

  /**
   * Clears all stored logs
   */
  clearLogs(): void {
    this.logs = [];
  }

  /**
   * Enables remote logging
   */
  enableRemoteLogging(): void {
    this.remoteLoggingEnabled = true;
  }

  /**
   * Disables remote logging
   */
  disableRemoteLogging(): void {
    this.remoteLoggingEnabled = false;
  }

  /**
   * Creates a child logger with context
   */
  child(context: Record<string, any>): ContextLogger {
    return new ContextLogger(this, context);
  }
}

/**
 * Context logger that adds context to all logs
 */
class ContextLogger {
  constructor(
    private parent: Logger,
    private context: Record<string, any>
  ) {}

  debug(message: string, data?: any): void {
    this.parent.debug(message, { ...this.context, ...data });
  }

  info(message: string, data?: any): void {
    this.parent.info(message, { ...this.context, ...data });
  }

  warn(message: string, data?: any): void {
    this.parent.warn(message, { ...this.context, ...data });
  }

  error(message: string, error?: Error | any): void {
    this.parent.error(message, error instanceof Error ? error : { ...this.context, ...error });
  }
}

// Export singleton instance
export const logger = new Logger();

// Export convenience functions
export const log = {
  debug: (message: string, data?: any) => logger.debug(message, data),
  info: (message: string, data?: any) => logger.info(message, data),
  warn: (message: string, data?: any) => logger.warn(message, data),
  error: (message: string, error?: Error | any) => logger.error(message, error),
};

export default logger;

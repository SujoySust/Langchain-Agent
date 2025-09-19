import { LangChainConfig, LogLevel } from "../types";

/**
 * Logging System
 * Provides structured logging with configurable levels and output
 */
export class Logger {
  private config: LangChainConfig;

  constructor(config: LangChainConfig) {
    this.config = config;
  }

  private log(level: string, message: string, data?: any): void {
    if (!this.config.logging.enableConsole) return;

    const timestamp = new Date().toISOString();
    const logMessage = `[${timestamp}] [${level.toUpperCase()}] ${message}`;

    if (data) {
      console.log(logMessage, data);
    } else {
      console.log(logMessage);
    }
  }

  private shouldLog(level: LogLevel): boolean {
    const levels: LogLevel[] = ["debug", "info", "warn", "error"];
    const currentLevelIndex = levels.indexOf(this.config.logging.level);
    const messageLevelIndex = levels.indexOf(level);

    return messageLevelIndex >= currentLevelIndex;
  }

  debug(message: string, data?: any): void {
    if (this.shouldLog("debug")) {
      this.log("debug", message, data);
    }
  }

  info(message: string, data?: any): void {
    if (this.shouldLog("info")) {
      this.log("info", message, data);
    }
  }

  warn(message: string, data?: any): void {
    if (this.shouldLog("warn")) {
      this.log("warn", message, data);
    }
  }

  error(message: string, data?: any): void {
    if (this.shouldLog("error")) {
      this.log("error", message, data);
    }
  }

  setLogLevel(level: LogLevel): void {
    this.config.logging.level = level;
  }

  enableConsoleLogging(enable: boolean): void {
    this.config.logging.enableConsole = enable;
  }
}

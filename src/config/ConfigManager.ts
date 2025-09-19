import "dotenv/config";
import { LangChainConfig } from "../types";

/**
 * Configuration Manager
 * Handles loading, validation, and management of application configuration
 */
export class ConfigManager {
  private static instance: ConfigManager;
  private config: LangChainConfig;

  private constructor() {
    this.config = this.loadConfig();
    this.validateConfig();
  }

  public static getInstance(): ConfigManager {
    if (!ConfigManager.instance) {
      ConfigManager.instance = new ConfigManager();
    }
    return ConfigManager.instance;
  }

  private loadConfig(): LangChainConfig {
    return {
      openai: {
        apiKey: process.env.OPENAI_API_KEY || "",
        model: process.env.OPENAI_MODEL || "gpt-3.5-turbo",
        temperature: parseFloat(process.env.OPENAI_TEMPERATURE || "0.7"),
        maxTokens: parseInt(process.env.OPENAI_MAX_TOKENS || "1000"),
      },
      embeddings: {
        model: process.env.EMBEDDING_MODEL || "text-embedding-ada-002",
        chunkSize: parseInt(process.env.CHUNK_SIZE || "1000"),
        chunkOverlap: parseInt(process.env.CHUNK_OVERLAP || "200"),
      },
      logging: {
        level: (process.env.LOG_LEVEL as any) || "info",
        enableConsole: process.env.ENABLE_CONSOLE_LOGGING !== "false",
      },
    };
  }

  private validateConfig(): void {
    if (!this.config.openai.apiKey) {
      throw new Error("OPENAI_API_KEY environment variable is required");
    }
  }

  public getConfig(): LangChainConfig {
    return this.config;
  }

  public updateConfig(newConfig: Partial<LangChainConfig>): void {
    this.config = { ...this.config, ...newConfig };
    this.validateConfig();
  }
}

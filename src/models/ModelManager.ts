import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { LangChainConfig } from "../types";
import { Logger } from "../utils/Logger";

/**
 * Model Management
 * Handles initialization and management of LangChain models
 */
export class ModelManager {
  private config: LangChainConfig;
  private logger: Logger;
  private initialized: boolean = false;
  private chatModel: ChatOpenAI | null = null;
  private embeddings: OpenAIEmbeddings | null = null;
  private textSplitter: RecursiveCharacterTextSplitter | null = null;

  constructor(config: LangChainConfig, logger: Logger) {
    this.config = config;
    this.logger = logger;
  }

  async initialize(): Promise<void> {
    if (this.initialized) {
      this.logger.warn("ModelManager already initialized");
      return;
    }

    try {
      // Initialize ChatOpenAI model
      this.chatModel = new ChatOpenAI({
        openAIApiKey: this.config.openai.apiKey,
        modelName: this.config.openai.model,
        temperature: this.config.openai.temperature,
        maxTokens: this.config.openai.maxTokens,
      });

      // Initialize OpenAI Embeddings
      this.embeddings = new OpenAIEmbeddings({
        openAIApiKey: this.config.openai.apiKey,
        modelName: this.config.embeddings.model,
      });

      // Initialize Text Splitter
      this.textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: this.config.embeddings.chunkSize,
        chunkOverlap: this.config.embeddings.chunkOverlap,
      });

      this.initialized = true;
    } catch (error) {
      this.logger.error("Failed to initialize ModelManager", error);
      throw error;
    }
  }

  async testConnection(): Promise<boolean> {
    try {
      this.logger.info("Testing model connection...");

      if (!this.initialized || !this.chatModel) {
        throw new Error(
          "ModelManager must be initialized before testing connection"
        );
      }

      // Test the chat model with a simple completion
      const response = await this.chatModel.invoke(
        "Hello, this is a connection test."
      );

      if (response && response.content) {
        this.logger.info("Model connection test successful");
        return true;
      } else {
        throw new Error("Invalid response from chat model");
      }
    } catch (error) {
      this.logger.error("Model connection test failed", error);
      return false;
    }
  }

  getConfig(): LangChainConfig {
    return this.config;
  }

  isInitialized(): boolean {
    return this.initialized;
  }

  async reinitialize(): Promise<void> {
    this.initialized = false;
    this.chatModel = null;
    this.embeddings = null;
    this.textSplitter = null;
    await this.initialize();
  }

  // LangChain model integration methods
  getChatModel(): ChatOpenAI {
    if (!this.initialized) {
      throw new Error("ModelManager must be initialized before use");
    }

    if (!this.chatModel) {
      throw new Error("Chat model not initialized");
    }

    this.logger.debug("Getting chat model instance");
    return this.chatModel;
  }

  getEmbeddings(): OpenAIEmbeddings {
    if (!this.initialized) {
      throw new Error("ModelManager must be initialized before use");
    }

    if (!this.embeddings) {
      throw new Error("Embeddings model not initialized");
    }

    this.logger.debug("Getting embeddings instance");
    return this.embeddings;
  }

  getTextSplitter(): RecursiveCharacterTextSplitter {
    if (!this.initialized) {
      throw new Error("ModelManager must be initialized before use");
    }

    if (!this.textSplitter) {
      throw new Error("Text splitter not initialized");
    }

    this.logger.debug("Getting text splitter instance");
    return this.textSplitter;
  }
}

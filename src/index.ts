/**
 * LangChain Starter Template - Main Entry Point
 * A comprehensive starter template for LangChain R&D projects
 */

// Import types
import { LangChainConfig } from "./types";

// Import core modules
import { ConfigManager } from "./config";
import { LangChainCore } from "./core";
import { ModelManager } from "./models";
import { Logger } from "./utils";

// =============================================================================
// MAIN APPLICATION CLASS
// =============================================================================

/**
 * Main LangChain Starter Application
 * Orchestrates all components and provides a unified interface
 */
class LangChainStarter {
  private config: LangChainConfig;
  private logger: Logger;
  private modelManager: ModelManager;
  private core: LangChainCore;
  private initialized: boolean = false;

  constructor() {
    this.config = ConfigManager.getInstance().getConfig();
    this.logger = new Logger(this.config);
    this.modelManager = new ModelManager(this.config, this.logger);
    this.core = new LangChainCore(this.modelManager, this.logger);
  }

  async initialize(): Promise<void> {
    if (this.initialized) {
      this.logger.warn("LangChain starter already initialized");
      return;
    }

    try {
      this.logger.info("Initializing LangChain starter...");

      // Initialize model manager
      await this.modelManager.initialize();

      // Test connection
      const connectionOk = await this.modelManager.testConnection();
      if (!connectionOk) {
        throw new Error("Failed to establish connection with models");
      }

      this.initialized = true;
      this.logger.info("LangChain starter initialized successfully");
    } catch (error) {
      this.logger.error("Failed to initialize LangChain starter", error);
      throw error;
    }
  }

  getCore(): LangChainCore {
    if (!this.initialized) {
      throw new Error("LangChain starter must be initialized before use");
    }
    return this.core;
  }

  getModelManager(): ModelManager {
    return this.modelManager;
  }

  getLogger(): Logger {
    return this.logger;
  }

  getConfig(): LangChainConfig {
    return this.config;
  }

  isInitialized(): boolean {
    return this.initialized;
  }

  async reinitialize(): Promise<void> {
    this.initialized = false;
    await this.initialize();
  }
}

// =============================================================================
// EXAMPLE USAGE & DEMONSTRATIONS
// =============================================================================

async function runExamples() {
  const app = new LangChainStarter();

  try {
    // Initialize the application
    await app.initialize();
    const core = app.getCore();
    const logger = app.getLogger();

    logger.info("=== Running LangChain Starter Examples ===");

    // Example 1: Simple Chat
    logger.info("Example 1: Simple Chat");
    const chatResponse = await core.simpleChat(
      "What is artificial intelligence?",
      "You are a helpful AI assistant that provides concise explanations."
    );
    console.log("Chat Response:", chatResponse);

    // Example 2: Prompt Chain
    logger.info("Example 2: Prompt Chain");
    const promptChain = await core.createPromptChain(
      "Translate the following text to {language}: {text}"
    );
    const translation = await promptChain.call({
      language: "Spanish",
      text: "Hello, how are you?",
    });
    console.log("Translation:", translation);

    // Example 3: RAG with sample documents
    logger.info("Example 3: RAG (Retrieval-Augmented Generation)");
    const sampleDocs = [
      "LangChain is a framework for developing applications powered by language models.",
      "It enables applications that are context-aware and can reason about their environment.",
      "LangChain provides tools for prompt management, chains, agents, and memory.",
      "Vector databases are used to store and retrieve semantic information efficiently.",
    ];

    const ragChain = await core.createRAGChain(sampleDocs);
    const ragResponse = await ragChain.call({
      query: "What tools does LangChain provide?",
    });
    console.log("RAG Response:", ragResponse);

    // Example 4: Similarity Search
    logger.info("Example 4: Similarity Search");
    const similarDocs = await core.similaritySearch(
      "vector database",
      sampleDocs,
      2
    );
    console.log("Similar Documents:", similarDocs);

    // Example 5: Text Embeddings
    logger.info("Example 5: Text Embeddings");
    const embeddings = await core.getEmbeddings("Hello, world!");
    console.log("Embeddings length:", embeddings.length);

    // Example 6: Text Splitting
    logger.info("Example 6: Text Splitting");
    const longText =
      "This is a very long text that needs to be split into smaller chunks for processing. ".repeat(
        50
      );
    const chunks = await core.splitText(longText);
    console.log("Number of chunks:", chunks.length);

    logger.info("=== Examples completed successfully ===");
  } catch (error) {
    app.getLogger().error("Error running examples", error);
  }
}

// =============================================================================
// MAIN EXECUTION
// =============================================================================

async function main() {
  console.log("🚀 LangChain Starter Template");
  console.log("=============================");

  // Check if we should run examples
  if (process.env.RUN_EXAMPLES === "true") {
    await runExamples();
  } else {
    console.log("To run examples, set RUN_EXAMPLES=true in your .env file");
    console.log(
      "For custom usage, import the classes and create your own implementation"
    );

    // Basic initialization example
    try {
      const app = new LangChainStarter();
      await app.initialize();
      console.log("✅ LangChain starter initialized and ready for use!");

      // Show configuration (without sensitive data)
      const config = app.getConfig();
      console.log("📋 Current Configuration:");
      console.log(`  - Model: ${config.openai.model}`);
      console.log(`  - Temperature: ${config.openai.temperature}`);
      console.log(`  - Max Tokens: ${config.openai.maxTokens}`);
      console.log(`  - Embedding Model: ${config.embeddings.model}`);
      console.log(`  - Log Level: ${config.logging.level}`);
    } catch (error) {
      console.error("❌ Failed to initialize LangChain starter:", error);
      process.exit(1);
    }
  }
}

// Run main function if this file is executed directly
if (require.main === module) {
  main().catch((error) => {
    console.error("Fatal error:", error);
    process.exit(1);
  });
}

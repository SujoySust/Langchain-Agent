# LangChain Starter Template 🚀

A comprehensive, modular starter template for LangChain R&D projects. This template provides a well-structured foundation for building applications powered by language models, with clean separation of concerns and enterprise-ready architecture.

## ✨ Features

- 🏗️ **Modular Architecture** - Organized into separate modules for maintainability
- ⚙️ **Configuration Management** - Environment-based configuration with validation
- 📝 **Advanced Logging** - Structured logging with configurable levels
- 🤖 **Model Management** - Centralized model initialization and connection testing
- ⚡ **Core LangChain Features** - Pre-built chains, RAG, vector stores, and more
- 🔧 **TypeScript Support** - Full type safety and excellent IDE experience
- 📚 **Comprehensive Examples** - Ready-to-run demonstrations
- 🧪 **Testing Ready** - Modular design perfect for unit testing
- 📖 **Extensive Documentation** - Clear setup and usage instructions

## 🏛️ Architecture Overview

```
src/
├── index.ts              # Main entry point & orchestration
├── types/                # Type definitions & interfaces
├── config/               # Configuration management
├── utils/                # Utilities & logging
├── models/               # Model initialization & management
└── core/                 # Core LangChain functionalities
```

### 📦 Core Modules

| Module               | Purpose                                | Key Features                                     |
| -------------------- | -------------------------------------- | ------------------------------------------------ |
| **ConfigManager**    | Environment & configuration management | Singleton pattern, validation, hot-reload        |
| **Logger**           | Structured logging system              | Configurable levels, timestamps, console control |
| **ModelManager**     | LangChain model management             | Initialization, connection testing, lazy loading |
| **LangChainCore**    | Core LangChain functionalities         | Chains, RAG, embeddings, vector stores           |
| **LangChainStarter** | Main orchestration class               | Unified interface, lifecycle management          |

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or use this template
git clone <repository-url>
cd langchain-starter

# Install dependencies
yarn install
# or
npm install
```

### 2. Environment Setup

Create a `.env` file in the root directory:

```env
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (with defaults)
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=1000
EMBEDDING_MODEL=text-embedding-ada-002
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
LOG_LEVEL=info
ENABLE_CONSOLE_LOGGING=true
RUN_EXAMPLES=false
```

### 3. Build & Run

```bash
# Development mode
yarn dev

# Build for production
yarn build

# Run built version
yarn start
```

## 💡 Usage Examples

### Basic Usage with Main Class

```typescript
import { LangChainStarter } from "./src/index";

async function main() {
  // Initialize the application
  const app = new LangChainStarter();
  await app.initialize();

  // Get core functionality
  const core = app.getCore();

  // Simple chat
  const response = await core.simpleChat(
    "What is artificial intelligence?",
    "You are a helpful AI assistant."
  );

  console.log(response);
}
```

### Modular Usage with Individual Components

```typescript
import { ConfigManager } from "./src/config";
import { Logger } from "./src/utils";
import { ModelManager } from "./src/models";
import { LangChainCore } from "./src/core";

async function customSetup() {
  // Initialize components individually
  const config = ConfigManager.getInstance().getConfig();
  const logger = new Logger(config);
  const modelManager = new ModelManager(config, logger);
  const core = new LangChainCore(modelManager, logger);

  // Initialize models
  await modelManager.initialize();

  // Use core functionality
  const embeddings = await core.getEmbeddings("Hello world");
  console.log("Embeddings length:", embeddings.length);
}
```

### RAG (Retrieval-Augmented Generation) Example

```typescript
import { LangChainStarter } from "./src/index";

async function ragExample() {
  const app = new LangChainStarter();
  await app.initialize();
  const core = app.getCore();

  // Sample documents
  const documents = [
    "LangChain is a framework for developing applications powered by language models.",
    "It enables applications that are context-aware and can reason about their environment.",
    "LangChain provides tools for prompt management, chains, agents, and memory.",
  ];

  // Create RAG chain
  const ragChain = await core.createRAGChain(documents);

  // Query the chain
  const response = await ragChain.call({
    query: "What tools does LangChain provide?",
  });

  console.log("Answer:", response.result);
  console.log("Sources:", response.sourceDocuments);
}
```

### Vector Store & Similarity Search

```typescript
const core = app.getCore();

// Create vector store
const vectorStore = await core.createVectorStore(documents);

// Perform similarity search
const similarDocs = await core.similaritySearch(
  "machine learning framework",
  documents,
  3 // top 3 results
);

console.log("Similar documents:", similarDocs);
```

## 🔧 Configuration

The template uses a centralized configuration system:

```typescript
// Access configuration anywhere
import { ConfigManager } from "./src/config";

const config = ConfigManager.getInstance().getConfig();
console.log("Current model:", config.openai.model);

// Update configuration at runtime
ConfigManager.getInstance().updateConfig({
  openai: { temperature: 0.9 },
});
```

## 📊 Logging

Advanced logging with multiple levels:

```typescript
import { Logger } from "./src/utils";

const logger = new Logger(config);

logger.debug("Debug information", { data: "detailed info" });
logger.info("General information");
logger.warn("Warning message");
logger.error("Error occurred", error);

// Change log level at runtime
logger.setLogLevel("debug");
```

## 🧪 Examples & Demonstrations

Run the built-in examples:

```bash
# Set in .env file
RUN_EXAMPLES=true

# Then run
yarn dev
```

This will demonstrate:

- ✅ Simple chat completions
- ✅ Prompt chain creation
- ✅ RAG implementation
- ✅ Similarity search
- ✅ Text embeddings
- ✅ Document chunking

## 🛠️ Development

### Project Structure Details

```
src/
├── index.ts                    # Main orchestration & examples
├── types/
│   └── index.ts               # TypeScript definitions
├── config/
│   ├── index.ts              # Barrel export
│   └── ConfigManager.ts      # Configuration management
├── utils/
│   ├── index.ts              # Barrel export
│   └── Logger.ts             # Logging system
├── models/
│   ├── index.ts              # Barrel export
│   └── ModelManager.ts       # Model management
└── core/
    ├── index.ts              # Barrel export
    └── LangChainCore.ts      # Core LangChain features
```

### Adding New Features

1. **New Core Feature**: Add to `src/core/LangChainCore.ts`
2. **New Configuration**: Update `src/types/index.ts` and `src/config/ConfigManager.ts`
3. **New Utility**: Add to `src/utils/`
4. **New Model Type**: Extend `src/models/ModelManager.ts`

### TypeScript Integration

The template provides comprehensive type definitions:

```typescript
import {
  LangChainConfig,
  VectorStore,
  RAGChain,
  SearchResult,
} from "./src/types";

// Fully typed configuration
const config: LangChainConfig = {
  openai: {
    /* ... */
  },
  embeddings: {
    /* ... */
  },
  logging: {
    /* ... */
  },
};
```

## 🤝 Contributing

This template is designed to be extended and customized for your specific needs:

1. Fork the repository
2. Create your feature branch
3. Add your enhancements
4. Update documentation
5. Submit a pull request

## 📚 Documentation

- **[SETUP.md](./SETUP.md)** - Detailed setup and configuration guide
- **[Type Definitions](./src/types/index.ts)** - Complete TypeScript interfaces
- **[Examples](./src/index.ts)** - Comprehensive usage examples

## 🔮 Roadmap

Future enhancements planned:

- [ ] Integration with actual LangChain models
- [ ] Advanced agent implementations
- [ ] Memory management systems
- [ ] Streaming response support
- [ ] Custom tool integrations
- [ ] Performance monitoring
- [ ] Production deployment guides

## 📄 License

MIT License - feel free to use this template for your projects.

## 🆘 Support

For questions, issues, or feature requests:

1. Check the [SETUP.md](./SETUP.md) guide
2. Review the example code in `src/index.ts`
3. Open an issue in the repository

---

**Happy coding with LangChain! 🎉**

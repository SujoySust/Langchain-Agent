# LangChain Starter Template Setup

## Environment Configuration

Create a `.env` file in the root directory with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=1000

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-ada-002
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Logging Configuration
LOG_LEVEL=info
ENABLE_CONSOLE_LOGGING=true

# Example Configuration
RUN_EXAMPLES=false
```

## Getting Started

1. **Install Dependencies**

   ```bash
   yarn install
   # or
   npm install
   ```

2. **Set up Environment Variables**

   - Copy the environment configuration above to a `.env` file
   - Replace `your_openai_api_key_here` with your actual OpenAI API key

3. **Build the Project**

   ```bash
   yarn build
   # or
   npm run build
   ```

4. **Run the Application**

   ```bash
   yarn start
   # or
   npm start
   ```

5. **Development Mode**
   ```bash
   yarn dev
   # or
   npm run dev
   ```

## Usage Examples

### Basic Initialization

```typescript
import { LangChainStarter } from "./src/index";

const app = new LangChainStarter();
await app.initialize();
const core = app.getCore();
```

### Using Individual Modules

```typescript
// Import specific modules
import { ConfigManager } from "./src/config";
import { Logger } from "./src/utils";
import { ModelManager } from "./src/models";
import { LangChainCore } from "./src/core";

// Or import everything from main index
import {
  LangChainStarter,
  ConfigManager,
  Logger,
  ModelManager,
  LangChainCore,
} from "./src/index";

// Initialize individual components
const config = ConfigManager.getInstance().getConfig();
const logger = new Logger(config);
const modelManager = new ModelManager(config, logger);
const core = new LangChainCore(modelManager, logger);
```

### Simple Chat

```typescript
const response = await core.simpleChat(
  "What is artificial intelligence?",
  "You are a helpful AI assistant."
);
```

### RAG Implementation

```typescript
const documents = ["Document 1 content...", "Document 2 content..."];

const ragChain = await core.createRAGChain(documents);
const response = await ragChain.call({
  query: "Your question here",
});
```

## Features

- **Configuration Management**: Centralized config with environment variables
- **Model Initialization**: Automated setup for OpenAI models and embeddings
- **Core Functionalities**: Pre-built chains, RAG, and similarity search
- **Logging System**: Configurable logging with different levels
- **Error Handling**: Comprehensive error handling throughout
- **Example Usage**: Ready-to-run examples for common use cases

## Project Structure

```
src/
├── index.ts              # Main application entry point
├── types/
│   └── index.ts         # Type definitions and interfaces
├── config/
│   ├── index.ts         # Barrel export
│   └── ConfigManager.ts # Configuration management
├── utils/
│   ├── index.ts         # Barrel export
│   └── Logger.ts        # Logging system
├── models/
│   ├── index.ts         # Barrel export
│   └── ModelManager.ts  # Model initialization and management
├── core/
│   ├── index.ts         # Barrel export
│   └── LangChainCore.ts # Core LangChain functionalities
└── LangChainStarter     # Main orchestration class (in index.ts)
```

## Modular Architecture

The template is now organized into separate modules for better maintainability:

- **`types/`**: Type definitions and interfaces
- **`config/`**: Configuration management and environment setup
- **`utils/`**: Utility functions and logging
- **`models/`**: Model management and initialization
- **`core/`**: Core LangChain functionalities and chains

Each module has its own index.ts file for clean barrel exports.

/**
 * Type definitions for LangChain Starter Template
 */

export interface LangChainConfig {
  openai: {
    apiKey: string;
    model: string;
    temperature: number;
    maxTokens: number;
  };
  embeddings: {
    model: string;
    chunkSize: number;
    chunkOverlap: number;
  };
  logging: {
    level: "debug" | "info" | "warn" | "error";
    enableConsole: boolean;
  };
}

export interface VectorStore {
  documents: string[];
  search: (query: string, k?: number) => Promise<SearchResult[]>;
}

export interface SearchResult {
  content: string;
  score: number;
}

export interface RAGChain {
  vectorStore: VectorStore;
  call: (input: { query: string }) => Promise<RAGResponse>;
}

export interface RAGResponse {
  result: string;
  sourceDocuments: SearchResult[];
}

export interface PromptChain {
  template: string;
  call: (variables: Record<string, any>) => Promise<string>;
}

export type LogLevel = "debug" | "info" | "warn" | "error";

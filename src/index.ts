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
import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import {
  MemorySaver,
  MessagesAnnotation,
  StateGraph,
} from "@langchain/langgraph";
import { createReactAgent, ToolNode } from "@langchain/langgraph/prebuilt";
import { AIMessage, HumanMessage } from "@langchain/core/messages";

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

  constructor() {
    this.config = ConfigManager.getInstance().getConfig();
    this.logger = new Logger(this.config);
    this.modelManager = new ModelManager(this.config, this.logger);
  }

  async testAgent(): Promise<void> {
    try {
      // Define tools for the agent
      const tools = [
        new TavilySearchResults({
          apiKey: process.env.TAVILY_API_KEY || "",
          maxResults: 3,
        }),
      ];

      const toolNode = new ToolNode(tools);
      await this.modelManager.initialize();
      const model = this.modelManager.getChatModel().bindTools(tools);

      function shouldContinue({ messages }: typeof MessagesAnnotation.State) {
        const lastMessage = messages[messages.length - 1] as AIMessage;
        if (lastMessage.tool_calls?.length) {
          return "tools";
        }
        return "__end__";
      }

      async function callModel(state: typeof MessagesAnnotation.State) {
        const response = await model.invoke(state.messages);
        return {
          messages: [response],
        };
      }

      const workflow = new StateGraph(MessagesAnnotation)
        .addNode("agent", callModel)
        .addEdge("__start__", "agent")
        .addNode("tools", toolNode)
        .addEdge("tools", "agent")
        .addConditionalEdges("agent", shouldContinue);

      const app = workflow.compile();
      const response = await app.invoke({
        messages: [new HumanMessage("What is the weather in Khulna?")],
      });
      console.log(
        "Response:",
        response.messages[response.messages.length - 1].content
      );

      const nextResponse = await app.invoke({
        messages: [...response.messages, new HumanMessage("What about Dhaka?")],
      });
      console.log(
        "Response:",
        nextResponse.messages[nextResponse.messages.length - 1].content
      );
    } catch (error) {
      this.logger.error("Failed to initialize LangChain starter", error);
      throw error;
    }
  }
}

// =============================================================================
// MAIN EXECUTION
// =============================================================================

async function main() {
  try {
    const app = new LangChainStarter();
    await app.testAgent();
  } catch (error) {
    process.exit(1);
  }
}

// Run main function if this file is executed directly
if (require.main === module) {
  main().catch((error) => {
    console.error("Fatal error:", error);
    process.exit(1);
  });
}

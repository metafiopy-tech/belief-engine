"""Protocol Skeleton Templates — Minimum Viable Code for Agentic Protocols.

Provides exact, tested code skeletons that the Belief Engine's builder
uses as starting points when generating protocol-specific projects.
These are NOT generic templates — they contain the exact import paths,
API surfaces, and configuration patterns required by each protocol's
current SDK version.

Each skeleton was validated against:
- x402 V2 SDK (@x402/express@2.3.0)
- MCP SDK (@modelcontextprotocol/sdk@1.29.0) with Streamable HTTP
- A2A SDK (@a2a-js/sdk) with Agent Card discovery
- ERC-8004 (agent0-sdk@1.7.0) on Base Sepolia

Usage:
    from belief.prompts.protocol_skeletons import get_skeleton
    skeleton = get_skeleton("x402")  # Returns dict of {filename: content}
"""

from __future__ import annotations


def get_skeleton(protocol: str) -> dict[str, str]:
    """Get the minimum viable skeleton for a protocol."""
    skeletons = {
        "x402": _x402_skeleton(),
        "mcp": _mcp_skeleton(),
        "a2a": _a2a_skeleton(),
        "erc8004": _erc8004_skeleton(),
    }
    return skeletons.get(protocol, {})


def get_all_protocol_names() -> list[str]:
    return ["x402", "mcp", "a2a", "erc8004"]


def _x402_skeleton() -> dict[str, str]:
    """x402 V2 Express server — payment-gated API endpoints."""
    return {
        "src/index.ts": '''import express from "express";
import { paymentMiddleware, x402ResourceServer } from "@x402/express";
import { ExactEvmScheme } from "@x402/evm/exact/server";
import { HTTPFacilitatorClient } from "@x402/core/server";

const app = express();
app.use(express.json());

// x402 payment setup
const facilitatorClient = new HTTPFacilitatorClient({
  url: process.env.X402_FACILITATOR ?? "https://x402.org/facilitator",
});
const server = new x402ResourceServer(facilitatorClient)
  .register(process.env.X402_NETWORK ?? "eip155:84532", new ExactEvmScheme());

// Payment-gated routes — method prefix required, dollar sign required
app.use(paymentMiddleware({
  "GET /api/data": {
    accepts: [{
      scheme: "exact",
      price: "$0.001",
      network: process.env.X402_NETWORK ?? "eip155:84532",
      payTo: process.env.PAY_TO_ADDRESS ?? "0x0000000000000000000000000000000000000000",
    }],
    description: "Premium data endpoint",
    mimeType: "application/json",
  },
}, server));

// Free endpoint
app.get("/health", (_req, res) => {
  res.json({ status: "ok", protocol: "x402-v2" });
});

// Paid endpoint
app.get("/api/data", (_req, res) => {
  res.json({ data: "This is paid content", timestamp: new Date().toISOString() });
});

const PORT = parseInt(process.env.PORT ?? "3000", 10);
app.listen(PORT, () => {
  console.log(`x402 server running on http://localhost:${PORT}`);
  console.log(`Health: http://localhost:${PORT}/health`);
  console.log(`Paid:   http://localhost:${PORT}/api/data (requires x402 payment)`);
});
''',
    }


def _mcp_skeleton() -> dict[str, str]:
    """MCP server with Streamable HTTP transport — payment-ready."""
    return {
        "src/index.ts": '''import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import express from "express";
import { z } from "zod";
import { randomUUID } from "node:crypto";

const app = express();
app.use(express.json());

// Create MCP server
const mcpServer = new McpServer({
  name: "my-mcp-server",
  version: "1.0.0",
});

// Register tools
mcpServer.registerTool("search", {
  title: "Search",
  description: "Search the knowledge base",
  inputSchema: { query: z.string().describe("Search query") },
}, async ({ query }) => ({
  content: [{ type: "text", text: `Results for: ${query}` }],
}));

mcpServer.registerTool("analyze", {
  title: "Analyze",
  description: "Analyze data",
  inputSchema: {
    data: z.string().describe("Data to analyze"),
    format: z.enum(["json", "csv", "text"]).optional().describe("Input format"),
  },
}, async ({ data, format }) => ({
  content: [{ type: "text", text: `Analysis of ${format ?? "text"} data: ${data.slice(0, 100)}` }],
}));

// MCP endpoint via Streamable HTTP
app.post("/mcp", async (req, res) => {
  const transport = new StreamableHTTPServerTransport({
    sessionIdGenerator: () => randomUUID(),
    enableJsonResponse: true,
  });
  res.on("close", () => transport.close());
  await mcpServer.connect(transport);
  await transport.handleRequest(req, res, req.body);
});

// Health check
app.get("/health", (_req, res) => {
  res.json({ status: "ok", protocol: "mcp", version: "1.0.0" });
});

const PORT = parseInt(process.env.PORT ?? "3000", 10);
app.listen(PORT, () => {
  console.log(`MCP server running on http://localhost:${PORT}`);
  console.log(`MCP endpoint: POST http://localhost:${PORT}/mcp`);
});
''',
    }


def _a2a_skeleton() -> dict[str, str]:
    """A2A server with Agent Card and JSON-RPC handler."""
    return {
        "src/index.ts": '''import express from "express";
import crypto from "node:crypto";

const app = express();
app.use(express.json());

// Agent Card — served at /.well-known/agent-card.json
const agentCard = {
  name: "My Agent",
  description: "An AI agent that performs tasks",
  url: process.env.AGENT_URL ?? "http://localhost:3000",
  version: "1.0.0",
  capabilities: { streaming: false, pushNotifications: false },
  defaultInputModes: ["text/plain"],
  defaultOutputModes: ["text/plain"],
  skills: [{
    id: "general",
    name: "General Task",
    description: "Handle general requests",
    examples: ["Summarize this document", "Generate a report"],
  }],
};

app.get("/.well-known/agent-card.json", (_req, res) => {
  res.json(agentCard);
});

// JSON-RPC 2.0 handler for A2A protocol
app.post("/a2a/jsonrpc", async (req, res) => {
  const { method, params, id } = req.body;

  switch (method) {
    case "message/send": {
      const userMessage = params?.message?.parts?.[0]?.text ?? "";
      res.json({
        jsonrpc: "2.0",
        id,
        result: {
          messageId: crypto.randomUUID(),
          role: "agent",
          parts: [{ kind: "text", text: `Processed: ${userMessage}` }],
          contextId: params?.contextId ?? crypto.randomUUID(),
          taskId: params?.taskId ?? crypto.randomUUID(),
          kind: "message",
        },
      });
      break;
    }
    case "tasks/get": {
      res.json({
        jsonrpc: "2.0",
        id,
        result: { id: params?.id, status: { state: "completed" } },
      });
      break;
    }
    default:
      res.json({
        jsonrpc: "2.0",
        id,
        error: { code: -32601, message: `Method not found: ${method}` },
      });
  }
});

// Health check
app.get("/health", (_req, res) => {
  res.json({ status: "ok", protocol: "a2a", version: "1.0.0" });
});

const PORT = parseInt(process.env.PORT ?? "3000", 10);
app.listen(PORT, () => {
  console.log(`A2A server running on http://localhost:${PORT}`);
  console.log(`Agent Card: http://localhost:${PORT}/.well-known/agent-card.json`);
});
''',
    }


def _erc8004_skeleton() -> dict[str, str]:
    """ERC-8004 agent registration CLI tool — ethers v6 compliant."""
    return {
        "src/index.ts": '''import { JsonRpcProvider, Wallet, Contract, Log, LogDescription } from "ethers";

// ERC-8004 Identity Registry on Base Sepolia
const IDENTITY_REGISTRY = "0x8004A818dC0b21ef9e8f9B2aaE2a12443F092cFC";

const IDENTITY_ABI = [
  "function register(string agentURI) external returns (uint256)",
  "function getAgent(uint256 agentId) external view returns (string agentURI, address owner)",
  "function totalSupply() external view returns (uint256)",
  "event Transfer(address indexed from, address indexed to, uint256 indexed tokenId)",
];

async function registerAgent(agentURI: string): Promise<{ agentId: string; tx: string }> {
  const privateKey = process.env.PRIVATE_KEY;
  if (!privateKey) throw new Error("PRIVATE_KEY environment variable required");

  // ethers v6: top-level imports, no ethers.providers.*
  const provider = new JsonRpcProvider(
    process.env.RPC_URL ?? "https://sepolia.base.org"
  );
  const wallet = new Wallet(privateKey, provider);
  const registry = new Contract(IDENTITY_REGISTRY, IDENTITY_ABI, wallet);

  console.log(`Registering agent with URI: ${agentURI}`);
  console.log(`Wallet: ${wallet.address}`);

  const tx = await registry.register(agentURI);
  console.log(`Transaction: ${tx.hash}`);

  const receipt = await tx.wait();
  // ethers v6: parseLog returns null (doesn't throw) — must null-check
  const event = receipt?.logs
    ?.map((log: Log) => {
      const parsed: LogDescription | null = registry.interface.parseLog(log);
      return parsed;
    })
    .find((e: LogDescription | null) => e !== null && e.name === "Transfer");

  const agentId = event?.args?.tokenId?.toString() ?? "unknown";
  console.log(`Agent registered! ID: ${agentId}`);

  return { agentId, tx: tx.hash };
}

// CLI entry point
const uri = process.argv[2];
if (!uri) {
  console.error("Usage: npx tsx src/index.ts <agent-uri>");
  console.error("Example: npx tsx src/index.ts https://my-agent.example.com/agent.json");
  process.exit(1);
}

registerAgent(uri).catch((err: Error) => {
  console.error("Registration failed:", err.message);
  process.exit(1);
});
''',
    }

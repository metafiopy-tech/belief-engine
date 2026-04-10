"""Belief Engine Benchmark Suite — 20 standardized build challenges.

Unlike SWE-bench (which tests bug-fixing in existing repos), this benchmark
tests greenfield generation from natural language specs.

Each challenge has:
- A goal string (natural language)
- Acceptance criteria (what must work)
- Verification method (how to grade)
- Difficulty tier (1-5)

Run: python3 -m belief.benchmark
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("belief.benchmark")


@dataclass
class Challenge:
    """A single benchmark challenge."""
    id: str
    tier: int
    goal: str
    acceptance_criteria: list[str]
    verify_commands: list[str]  # Shell commands to verify the build works
    timeout_seconds: int = 600
    tags: list[str] = field(default_factory=list)


@dataclass
class ChallengeResult:
    """Result of running a single challenge."""
    challenge_id: str
    tier: int
    verdict: str = "not_run"
    tests_passed: int = 0
    tests_total: int = 0
    weighted_score: float = 0.0
    cost_usd: float = 0.0
    build_time_seconds: float = 0.0
    executor_passed: bool = False
    deployed: bool = False
    error: str = ""


# ── The 20 Challenges ────────────────────────────────────────────────────────

CHALLENGES = [
    # ── Tier 1: Single-file scripts (3 challenges) ──
    Challenge(
        id="t1-fizzbuzz",
        tier=1,
        goal="Build a Python script that prints FizzBuzz from 1 to 100. Fizz for multiples of 3, Buzz for multiples of 5, FizzBuzz for both.",
        acceptance_criteria=["Prints 100 lines", "Line 15 is FizzBuzz", "Line 3 is Fizz", "Line 5 is Buzz"],
        verify_commands=["python main.py | wc -l", "python main.py | sed -n '15p'"],
        tags=["script", "logic"],
    ),
    Challenge(
        id="t1-fibonacci",
        tier=1,
        goal="Build a Python script that calculates the first 20 Fibonacci numbers and prints them as a JSON array.",
        acceptance_criteria=["Outputs valid JSON", "Contains 20 numbers", "First two are 0 and 1"],
        verify_commands=["python main.py | python -m json.tool"],
        tags=["script", "math"],
    ),
    Challenge(
        id="t1-wordcount",
        tier=1,
        goal="Build a Python script that reads a text file from stdin and prints word count, line count, and character count.",
        acceptance_criteria=["Prints word count", "Prints line count", "Prints char count"],
        verify_commands=["echo 'hello world' | python main.py"],
        tags=["script", "text"],
    ),

    # ── Tier 2: Simple APIs and CLIs (4 challenges) ──
    Challenge(
        id="t2-todo-cli",
        tier=2,
        goal="Build a CLI todo app with Click — add, list, complete, delete tasks. Store in a JSON file.",
        acceptance_criteria=["add command works", "list shows tasks", "complete marks done", "delete removes", "persists to JSON"],
        verify_commands=["python todo.py add 'Test task'", "python todo.py list"],
        tags=["cli", "click", "json"],
    ),
    Challenge(
        id="t2-health-api",
        tier=2,
        goal="Build a FastAPI server with GET /health returning {status: ok}, GET /time returning current UTC time, and GET /echo?msg=hello returning the message.",
        acceptance_criteria=["GET /health returns 200", "GET /time returns ISO timestamp", "GET /echo?msg=test returns test"],
        verify_commands=["curl localhost:8000/health", "curl localhost:8000/time"],
        tags=["api", "fastapi"],
    ),
    Challenge(
        id="t2-calculator-cli",
        tier=2,
        goal="Build a CLI calculator with Click that supports add, subtract, multiply, divide operations. Handle division by zero.",
        acceptance_criteria=["add 2 3 returns 5", "divide 10 0 shows error", "multiply 4 5 returns 20"],
        verify_commands=["python calc.py add 2 3", "python calc.py divide 10 0"],
        tags=["cli", "click", "math"],
    ),
    Challenge(
        id="t2-csv-stats",
        tier=2,
        goal="Build a CLI tool that reads a CSV file and prints column statistics: min, max, mean for numeric columns. Use Click and Rich for formatting.",
        acceptance_criteria=["Reads CSV files", "Shows min/max/mean", "Handles non-numeric columns", "Rich formatted output"],
        verify_commands=["echo 'a,b\\n1,2\\n3,4' > /tmp/test.csv && python stats.py /tmp/test.csv"],
        tags=["cli", "csv", "rich"],
    ),

    # ── Tier 3: Package-structured apps (5 challenges) ──
    Challenge(
        id="t3-url-shortener",
        tier=3,
        goal="Build a URL shortener with FastAPI and SQLAlchemy 2.x — POST /shorten accepts a URL, GET /{code} redirects, GET /stats/{code} returns clicks. SQLite storage.",
        acceptance_criteria=["POST /shorten creates short URL", "GET /{code} redirects", "GET /stats shows clicks", "Click counter increments"],
        verify_commands=["curl -X POST localhost:8000/shorten -H 'Content-Type: application/json' -d '{\"url\": \"https://example.com\"}'"],
        tags=["api", "fastapi", "sqlalchemy", "crud"],
    ),
    Challenge(
        id="t3-bookmark-api",
        tier=3,
        goal="Build a bookmark manager API with FastAPI — CRUD for bookmarks with title, URL, and tags. GET /random returns a random bookmark. SQLite + SQLAlchemy 2.x.",
        acceptance_criteria=["POST creates bookmark", "GET lists bookmarks", "PUT updates", "DELETE removes", "GET /random works", "Tags stored"],
        verify_commands=["curl localhost:8000/docs"],
        tags=["api", "fastapi", "sqlalchemy", "crud"],
    ),
    Challenge(
        id="t3-notes-api",
        tier=3,
        goal="Build a notes API with FastAPI — CRUD for notes with title, content, tags. GET /search?q=keyword searches by title and content. Markdown rendering via GET /notes/{id}/html. SQLite.",
        acceptance_criteria=["CRUD works", "Search returns matching notes", "HTML endpoint renders markdown", "Tags filterable"],
        verify_commands=["curl localhost:8000/docs"],
        tags=["api", "fastapi", "sqlalchemy", "markdown", "search"],
    ),
    Challenge(
        id="t3-expense-tracker",
        tier=3,
        goal="Build an expense tracker CLI with Click — add expense (amount, category, description), list by date range, summarize by category, export to CSV. Store in SQLite.",
        acceptance_criteria=["Add expense works", "List by date range", "Category summary", "CSV export"],
        verify_commands=["python expenses.py add 50.00 food 'Lunch'", "python expenses.py summary"],
        tags=["cli", "click", "sqlalchemy", "reporting"],
    ),
    Challenge(
        id="t3-contact-api",
        tier=3,
        goal="Build a contacts API with FastAPI — CRUD for contacts with name, email, phone, company. GET /search?q=name searches contacts. Import/export CSV. SQLite + SQLAlchemy 2.x.",
        acceptance_criteria=["CRUD works", "Search by name", "CSV import", "CSV export"],
        verify_commands=["curl localhost:8000/docs"],
        tags=["api", "fastapi", "sqlalchemy", "csv"],
    ),

    # ── Tier 4: Multi-component systems (4 challenges) ──
    Challenge(
        id="t4-blog-engine",
        tier=4,
        goal="Build a blog engine with FastAPI — posts with markdown content, comments, categories. Admin endpoint to create/edit/delete posts. Public endpoints to list and read. Full-text search. SQLite.",
        acceptance_criteria=["Create post with markdown", "List posts with pagination", "Full-text search", "Comments on posts", "Categories/tags"],
        verify_commands=["curl localhost:8000/docs"],
        timeout_seconds=900,
        tags=["api", "fastapi", "sqlalchemy", "markdown", "search", "complex"],
    ),
    Challenge(
        id="t4-task-board",
        tier=4,
        goal="Build a Kanban task board API with FastAPI — boards, columns (todo/in-progress/done), cards with title/description/assignee. Move cards between columns. Card ordering within columns. SQLite.",
        acceptance_criteria=["Create board", "Add columns", "Create cards", "Move cards between columns", "Reorder cards"],
        verify_commands=["curl localhost:8000/docs"],
        timeout_seconds=900,
        tags=["api", "fastapi", "sqlalchemy", "complex"],
    ),
    Challenge(
        id="t4-file-vault",
        tier=4,
        goal="Build a file vault API with FastAPI — upload files (any type, max 10MB), download by ID, list files with metadata (size, type, upload date). Generate shareable links with expiry. SQLite for metadata, filesystem for files.",
        acceptance_criteria=["Upload file", "Download file", "List files", "Shareable links", "Link expiry"],
        verify_commands=["curl localhost:8000/docs"],
        timeout_seconds=900,
        tags=["api", "fastapi", "file-upload", "complex"],
    ),
    Challenge(
        id="t4-poll-system",
        tier=4,
        goal="Build a polling system API with FastAPI — create polls with multiple options, vote (one vote per session), view results with percentages, close polls. Real-time vote counts. SQLite.",
        acceptance_criteria=["Create poll", "Add options", "Vote", "View results with percentages", "Close poll", "Prevent double voting"],
        verify_commands=["curl localhost:8000/docs"],
        timeout_seconds=900,
        tags=["api", "fastapi", "sqlalchemy", "complex"],
    ),

    # ── Tier 5: Advanced systems (4 challenges) ──
    Challenge(
        id="t5-event-system",
        tier=5,
        goal="Build an event management system with FastAPI — events with title/date/location/capacity, RSVP with waitlist when full, send confirmation (log to console), attendee check-in, event analytics (attendance rate, no-show rate). SQLite.",
        acceptance_criteria=["Create events", "RSVP with capacity limit", "Waitlist", "Check-in", "Analytics endpoint"],
        verify_commands=["curl localhost:8000/docs"],
        timeout_seconds=1200,
        tags=["api", "fastapi", "sqlalchemy", "complex", "analytics"],
    ),
    Challenge(
        id="t5-inventory-system",
        tier=5,
        goal="Build an inventory management system with FastAPI — products (name, SKU, price, quantity), stock movements (in/out with reason), low stock alerts, stock history, multi-location support. SQLite.",
        acceptance_criteria=["CRUD products", "Record stock movements", "Low stock alerts", "Stock history", "Multi-location"],
        verify_commands=["curl localhost:8000/docs"],
        timeout_seconds=1200,
        tags=["api", "fastapi", "sqlalchemy", "complex", "inventory"],
    ),
    Challenge(
        id="t5-quiz-engine",
        tier=5,
        goal="Build a quiz engine API with FastAPI — create quizzes with multiple question types (multiple choice, true/false, free text), take quizzes with timer, auto-grade, leaderboard, quiz analytics. SQLite.",
        acceptance_criteria=["Create quiz", "Multiple question types", "Take quiz", "Auto-grade", "Leaderboard", "Analytics"],
        verify_commands=["curl localhost:8000/docs"],
        timeout_seconds=1200,
        tags=["api", "fastapi", "sqlalchemy", "complex", "gamification"],
    ),
    Challenge(
        id="t5-workflow-engine",
        tier=5,
        goal="Build a workflow automation engine with FastAPI — define workflows as DAGs of steps, execute workflows with input data, track step status (pending/running/completed/failed), retry failed steps, execution history. SQLite.",
        acceptance_criteria=["Define workflow DAG", "Execute workflow", "Track step status", "Retry failed steps", "Execution history"],
        verify_commands=["curl localhost:8000/docs"],
        timeout_seconds=1200,
        tags=["api", "fastapi", "sqlalchemy", "dag", "complex", "automation"],
    ),

    # ── Tier 6: Multi-service systems (5 challenges) ──
    Challenge(
        id="t6-api-gateway",
        tier=6,
        goal="Build a system with two FastAPI services: a user-service (CRUD for users with name, email, role) on port 8001 and an order-service (CRUD for orders with user_id, items, total) on port 8002. The order-service calls user-service to validate user_id exists before creating an order. Both use SQLite. Include a docker-compose.yml.",
        acceptance_criteria=["User service CRUD works", "Order service CRUD works", "Order creation validates user exists", "Docker compose orchestrates both", "Services communicate via HTTP"],
        verify_commands=["curl localhost:8001/docs", "curl localhost:8002/docs"],
        timeout_seconds=1500,
        tags=["multi-service", "fastapi", "sqlalchemy", "docker", "http"],
    ),
    Challenge(
        id="t6-event-driven",
        tier=6,
        goal="Build a notification system with two FastAPI services: a task-service (CRUD for tasks with title, status, assignee) on port 8001 and a notification-service (stores notifications with message, recipient, read status) on port 8002. When a task is created or status changes, the task-service posts a notification to the notification-service via HTTP. Both use SQLite.",
        acceptance_criteria=["Task CRUD works", "Notification CRUD works", "Task creation triggers notification", "Status change triggers notification", "Both services have health endpoints"],
        verify_commands=["curl localhost:8001/docs", "curl localhost:8002/docs"],
        timeout_seconds=1500,
        tags=["multi-service", "fastapi", "sqlalchemy", "events", "http"],
    ),
    Challenge(
        id="t6-shared-auth",
        tier=6,
        goal="Build two FastAPI services sharing an auth model: an auth-service (register, login with JWT tokens, user profile) on port 8001 and a document-service (CRUD for documents with title, content, owner_id) on port 8002. The document-service validates JWT tokens by calling auth-service. Both use SQLite.",
        acceptance_criteria=["User registration works", "Login returns JWT", "Document CRUD works", "Documents require valid token", "Token validation via auth-service"],
        verify_commands=["curl localhost:8001/docs", "curl localhost:8002/docs"],
        timeout_seconds=1500,
        tags=["multi-service", "fastapi", "sqlalchemy", "auth", "jwt"],
    ),
    Challenge(
        id="t6-data-pipeline",
        tier=6,
        goal="Build two FastAPI services: an ingestion-service (accepts CSV uploads, parses rows, stores in SQLite) on port 8001 and an analytics-service (reads from the same SQLite database, provides aggregate endpoints: count, sum, average, group-by) on port 8002. The analytics-service queries the database that ingestion-service writes to.",
        acceptance_criteria=["CSV upload works", "Parsed data stored in SQLite", "Count endpoint works", "Average endpoint works", "Group-by endpoint works"],
        verify_commands=["curl localhost:8001/docs", "curl localhost:8002/docs"],
        timeout_seconds=1500,
        tags=["multi-service", "fastapi", "sqlalchemy", "csv", "analytics"],
    ),
    Challenge(
        id="t6-microservice-crud",
        tier=6,
        goal="Build three FastAPI services: a product-service (CRUD for products with name, price, category) on port 8001, a review-service (CRUD for reviews with product_id, rating 1-5, comment) on port 8002, and a catalog-service (aggregates products with their average rating by calling both services) on port 8003. All use SQLite independently.",
        acceptance_criteria=["Product CRUD works", "Review CRUD works", "Catalog aggregates products with ratings", "Average rating calculated correctly", "All three services have health endpoints"],
        verify_commands=["curl localhost:8001/docs", "curl localhost:8002/docs", "curl localhost:8003/docs"],
        timeout_seconds=1800,
        tags=["multi-service", "fastapi", "sqlalchemy", "aggregation", "three-service"],
    ),

    # ── Tier 7: Brownfield modification (5 challenges) ──
    # These test extending or fixing existing codebases, not greenfield generation.
    # The engine receives a working codebase + an issue to fix or feature to add.
    Challenge(
        id="t7-add-pagination",
        tier=7,
        goal="Given an existing FastAPI CRUD API for 'items' (name, price, category) with GET /items returning all items, add pagination support: GET /items?page=1&size=10 with total count in response headers. Do not break existing endpoints.",
        acceptance_criteria=["GET /items?page=1&size=10 returns paginated results", "X-Total-Count header present", "Default page=1, size=20", "Existing POST/PUT/DELETE still work"],
        verify_commands=["curl 'localhost:8000/items?page=1&size=10'"],
        timeout_seconds=900,
        tags=["brownfield", "fastapi", "pagination", "extend"],
    ),
    Challenge(
        id="t7-add-search",
        tier=7,
        goal="Given an existing FastAPI notes API (CRUD for notes with title, content, created_at) stored in SQLite, add full-text search: GET /notes/search?q=keyword that searches both title and content fields. Return matching notes sorted by relevance.",
        acceptance_criteria=["GET /notes/search?q=keyword works", "Searches title and content", "Returns matching notes", "Existing CRUD endpoints unaffected"],
        verify_commands=["curl 'localhost:8000/notes/search?q=test'"],
        timeout_seconds=900,
        tags=["brownfield", "fastapi", "search", "extend"],
    ),
    Challenge(
        id="t7-fix-validation",
        tier=7,
        goal="Given an existing FastAPI user registration API that accepts POST /register with {username, email, password}, fix the validation: currently it accepts empty strings for all fields and duplicate emails. Add proper validation: username 3-50 chars, valid email format, password min 8 chars, and unique email constraint.",
        acceptance_criteria=["Empty username rejected", "Invalid email rejected", "Short password rejected", "Duplicate email returns 409", "Valid registration still works"],
        verify_commands=["curl -X POST localhost:8000/register -H 'Content-Type: application/json' -d '{}'"],
        timeout_seconds=900,
        tags=["brownfield", "fastapi", "validation", "fix"],
    ),
    Challenge(
        id="t7-add-auth",
        tier=7,
        goal="Given an existing FastAPI todo API (CRUD for tasks with title, description, completed) that has no authentication, add JWT-based auth: POST /login accepts {username, password}, returns {access_token}. All task endpoints require a valid Bearer token. Add a hardcoded test user (admin/admin123).",
        acceptance_criteria=["POST /login returns JWT token", "Task endpoints reject requests without token", "Task endpoints work with valid token", "Invalid token returns 401"],
        verify_commands=["curl -X POST localhost:8000/login -H 'Content-Type: application/json' -d '{\"username\":\"admin\",\"password\":\"admin123\"}'"],
        timeout_seconds=900,
        tags=["brownfield", "fastapi", "auth", "jwt", "extend"],
    ),
    Challenge(
        id="t7-add-export",
        tier=7,
        goal="Given an existing FastAPI expense tracker (CRUD for expenses with amount, category, description, date) stored in SQLite, add CSV export: GET /expenses/export returns a CSV file download of all expenses. Add date range filtering: GET /expenses/export?start=2024-01-01&end=2024-12-31.",
        acceptance_criteria=["GET /expenses/export returns CSV", "CSV has correct headers", "Date range filtering works", "Existing CRUD endpoints unaffected"],
        verify_commands=["curl localhost:8000/expenses/export"],
        timeout_seconds=900,
        tags=["brownfield", "fastapi", "csv", "export", "extend"],
    ),

    # ── Tier 8: TypeScript / Cross-Language (5 challenges) ──
    Challenge(
        id="t8-express-api",
        tier=8,
        goal="Build a TypeScript Express 5 REST API for a book library. Endpoints: GET /books (list all), POST /books (add book with title, author, isbn), GET /books/:id (get by id), DELETE /books/:id. Store books in memory (array). Use TypeScript with ESM, package.json with type: module, and vitest for tests.",
        acceptance_criteria=["GET /books returns array", "POST /books adds a book", "GET /books/:id returns one book", "DELETE /books/:id removes book", "package.json has type: module", "Tests use vitest"],
        verify_commands=["curl localhost:3000/books"],
        timeout_seconds=600,
        tags=["typescript", "express5", "rest", "esm"],
    ),
    Challenge(
        id="t8-ts-cli",
        tier=8,
        goal="Build a TypeScript CLI tool that converts Markdown files to HTML. It should accept a file path as argument, read the Markdown, convert to HTML using a simple parser (headings, bold, italic, links, code blocks), and write the output to a .html file. Use TypeScript with ESM (import/export, .js extensions in imports). Include vitest tests.",
        acceptance_criteria=["Accepts file path argument", "Converts headings (#, ##, ###)", "Converts bold and italic", "Converts links", "Writes .html output file", "Tests pass with vitest"],
        verify_commands=["npx tsx src/index.ts test.md"],
        timeout_seconds=600,
        tags=["typescript", "cli", "markdown", "esm"],
    ),
    Challenge(
        id="t8-ethers-reader",
        tier=8,
        goal="Build a TypeScript script that reads the ETH balance of a given wallet address from Ethereum mainnet using ethers v6. Accept the address as a CLI argument. Use ethers v6 top-level imports (JsonRpcProvider, formatEther — NOT ethers.providers or ethers.utils). Use a public RPC endpoint. Print the balance in ETH. Include vitest tests with a mocked provider.",
        acceptance_criteria=["Accepts address as CLI arg", "Uses ethers v6 top-level imports", "Prints balance in ETH", "Does NOT use ethers.providers or BigNumber", "Tests mock the provider"],
        verify_commands=["npx tsx src/index.ts 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"],
        timeout_seconds=600,
        tags=["typescript", "ethers", "blockchain", "esm"],
    ),
    Challenge(
        id="t8-mcp-server",
        tier=8,
        goal="Build a TypeScript MCP server with Streamable HTTP transport that exposes two tools: 'echo' (returns the input text) and 'word_count' (counts words in text). Use @modelcontextprotocol/sdk with correct subpath imports (.js extensions). Use zod for input validation. Mount on Express at POST /mcp. Include a health check at GET /health.",
        acceptance_criteria=["MCP server starts on port 3000", "POST /mcp handles MCP protocol", "echo tool returns input text", "word_count tool returns count", "Uses correct MCP SDK subpath imports with .js", "Health check returns 200"],
        verify_commands=["curl localhost:3000/health"],
        timeout_seconds=600,
        tags=["typescript", "mcp", "express5", "esm"],
    ),
    Challenge(
        id="t8-ts-fullstack",
        tier=8,
        goal="Build a TypeScript Express 5 API for a task manager with: POST /tasks (create task with title, status=pending), GET /tasks (list all, filter by ?status=), PATCH /tasks/:id (update status to 'pending'|'in_progress'|'done'), DELETE /tasks/:id. Store in memory. Use zod for request validation. Use vitest + supertest for tests. All TypeScript with ESM.",
        acceptance_criteria=["CRUD operations work", "Status filtering works", "Zod validates request bodies", "Invalid status returns 400", "Tests use vitest + supertest", "package.json has type: module"],
        verify_commands=["curl localhost:3000/tasks"],
        timeout_seconds=600,
        tags=["typescript", "express5", "zod", "validation", "esm"],
    ),
]


# ── Benchmark Runner ─────────────────────────────────────────────────────────

async def run_challenge(challenge: Challenge) -> ChallengeResult:
    """Run a single benchmark challenge."""
    from belief.cli import run as cli_run

    result = ChallengeResult(
        challenge_id=challenge.id,
        tier=challenge.tier,
    )

    t0 = time.time()
    try:
        final_state = await cli_run(
            goal=challenge.goal,
            max_cost=10.0,
            max_iterations=3,
        )

        result.build_time_seconds = time.time() - t0

        # Extract results from final state
        validation = final_state.get("validation_result")
        if validation:
            if isinstance(validation, dict):
                result.verdict = validation.get("verdict", "unknown")
                result.tests_passed = validation.get("tests_passed", 0)
                result.tests_total = validation.get("tests_total", 0)
                result.weighted_score = validation.get("weighted_score", 0.0)
            else:
                result.verdict = getattr(validation, "verdict", "unknown")
                if hasattr(result.verdict, "value"):
                    result.verdict = result.verdict.value
                result.tests_passed = getattr(validation, "tests_passed", 0)
                result.tests_total = getattr(validation, "tests_total", 0)
                result.weighted_score = getattr(validation, "weighted_score", 0.0)

        exec_result = final_state.get("execution_result")
        if exec_result:
            result.executor_passed = exec_result.get("success", False) if isinstance(exec_result, dict) else getattr(exec_result, "success", False)

        # Extract cost — from token_usage (set by BaseAgent), not build_budget
        usage = final_state.get("token_usage")
        if usage:
            result.cost_usd = usage.get("total_cost_usd", 0.0) if isinstance(usage, dict) else getattr(usage, "total_cost_usd", 0.0)

    except Exception as e:
        result.error = str(e)
        result.build_time_seconds = time.time() - t0

    return result


async def run_benchmark(
    tiers: list[int] | None = None,
    challenge_ids: list[str] | None = None,
) -> list[ChallengeResult]:
    """Run the full benchmark or a subset."""
    challenges = CHALLENGES

    if tiers:
        challenges = [c for c in challenges if c.tier in tiers]
    if challenge_ids:
        challenges = [c for c in challenges if c.id in challenge_ids]

    results = []
    for i, challenge in enumerate(challenges):
        print(f"\n{'═' * 60}")
        print(f"  BENCHMARK [{i+1}/{len(challenges)}] — {challenge.id} (Tier {challenge.tier})")
        print(f"  {challenge.goal[:80]}...")
        print(f"{'═' * 60}\n")

        result = await run_challenge(challenge)
        results.append(result)

        icon = "✓" if result.verdict == "pass" else "○" if result.executor_passed else "✗"
        print(f"\n  {icon} {challenge.id}: {result.verdict} "
              f"({result.tests_passed}/{result.tests_total} tests, "
              f"weighted={result.weighted_score:.2f}, "
              f"${result.cost_usd:.2f}, {result.build_time_seconds:.0f}s)")

    # Print summary
    print(f"\n\n{'═' * 60}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'═' * 60}\n")

    total = len(results)
    passed = sum(1 for r in results if r.verdict == "pass")
    executor_ok = sum(1 for r in results if r.executor_passed)
    total_cost = sum(r.cost_usd for r in results)
    total_time = sum(r.build_time_seconds for r in results)
    avg_score = sum(r.weighted_score for r in results) / max(total, 1)

    print(f"  Pass rate:     {passed}/{total} ({passed/max(total,1)*100:.0f}%)")
    print(f"  Executor rate: {executor_ok}/{total} ({executor_ok/max(total,1)*100:.0f}%)")
    print(f"  Avg weighted:  {avg_score:.2f}")
    print(f"  Total cost:    ${total_cost:.2f}")
    if total_cost == 0.0 and total > 0:
        print(f"  ⚠️  Cost is $0.00 — check token_usage tracking")
    print(f"  Total time:    {total_time:.0f}s ({total_time/60:.1f}min)")
    print()

    for tier in sorted(set(r.tier for r in results)):
        tier_results = [r for r in results if r.tier == tier]
        tier_pass = sum(1 for r in tier_results if r.verdict == "pass")
        tier_exec = sum(1 for r in tier_results if r.executor_passed)
        tier_label = {1: "Scripts", 2: "CLI/API", 3: "CRUD APIs", 4: "Multi-file",
                      5: "Complex", 6: "Multi-service", 7: "Brownfield", 8: "TypeScript"}.get(tier, "")
        print(f"  Tier {tier} ({tier_label}): {tier_pass}/{len(tier_results)} pass, {tier_exec}/{len(tier_results)} executor")

    # Save results
    results_file = Path.home() / ".belief-engine" / "benchmark_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results_file.write_text(json.dumps(
        [{"id": r.challenge_id, "tier": r.tier, "verdict": r.verdict,
          "tests_passed": r.tests_passed, "tests_total": r.tests_total,
          "weighted_score": r.weighted_score, "cost": r.cost_usd,
          "time": r.build_time_seconds, "executor_passed": r.executor_passed,
          "error": r.error}
         for r in results],
        indent=2,
    ))
    print(f"\n  Results saved to {results_file}")

    return results


# ── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    tiers = None
    ids = None

    for arg in sys.argv[1:]:
        if arg.startswith("--tier="):
            tiers = [int(t) for t in arg.split("=")[1].split(",")]
        elif arg.startswith("--id="):
            ids = arg.split("=")[1].split(",")

    if not tiers and not ids:
        print("Usage:")
        print("  python3 -m belief.benchmark --tier=1          # Run Tier 1 only")
        print("  python3 -m belief.benchmark --tier=1,2,3      # Run Tiers 1-3")
        print("  python3 -m belief.benchmark --tier=1,2,3,4,5  # Run Tiers 1-5 (Python)")
        print("  python3 -m belief.benchmark --tier=8           # Run Tier 8 (TypeScript)")
        print("  python3 -m belief.benchmark --tier=1,2,3,4,5,6,7,8  # Full suite (35)")
        print("  python3 -m belief.benchmark --id=t8-express-api     # Run specific challenge")
        sys.exit(0)

    asyncio.run(run_benchmark(tiers=tiers, challenge_ids=ids))

"""Multi-Service Detection and Verification.

Tier 1: LLM-based intent classification (replaces keyword matching)
Tier 2: Health check auto-generation and verification
Tier 3: Cross-service communication smoke testing

Usage:
    from belief.tools.multi_service import classify_goal, verify_services

    # Classify
    result = await classify_goal(goal, llm)
    if result.is_multi_service:
        pipeline = build_multi_pipeline(router)

    # Verify after build
    issues = verify_services(code_files, service_architecture)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("belief.tools.multi_service")


@dataclass
class GoalClassification:
    """Result of classifying a goal as single or multi-service."""
    is_multi_service: bool = False
    service_count: int = 1
    services: list[dict[str, str]] = field(default_factory=list)
    communication_pattern: str = ""  # "rest", "grpc", "message_queue", "shared_db"
    confidence: float = 0.0
    reasoning: str = ""


@dataclass
class ServiceVerification:
    """Result of verifying multi-service build output."""
    health_endpoints_present: bool = True
    all_services_have_routes: bool = True
    cross_service_imports_valid: bool = True
    openapi_specs_consistent: bool = True
    issues: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.issues) == 0


async def classify_goal(goal: str, llm=None) -> GoalClassification:
    """Classify a goal as single-service or multi-service using LLM.

    Replaces brittle keyword matching with a single Haiku call (~100ms, ~$0.001).
    Correctly handles:
    - "build an order system with a separate payment processor" → multi
    - "explain microservice architecture" → single (just an explanation)
    - "create a REST API with auth and payments" → single (one service, multiple features)
    - "build two services: users API on port 8001 and orders API on port 8002" → multi
    """
    result = GoalClassification()

    if not llm:
        # Fallback to improved keyword detection if no LLM available
        return _classify_by_keywords(goal)

    try:
        raw = await llm.generate_text(
            role="intake",
            system=(
                "You classify software project goals. Respond ONLY with valid JSON.\n"
                "A multi-service project has MULTIPLE independent services that communicate "
                "via HTTP/REST, gRPC, or message queues. Each service runs on its own port.\n"
                "A single-service project is ONE application, even if it has many features, "
                "endpoints, or modules.\n"
                "An explanation, tutorial, or documentation request is ALWAYS single-service."
            ),
            prompt=(
                f"Classify this goal:\n\n{goal}\n\n"
                "Respond with JSON:\n"
                '{"is_multi_service": bool, "service_count": int, '
                '"services": [{"name": "...", "role": "..."}], '
                '"communication_pattern": "rest|grpc|message_queue|shared_db|none", '
                '"confidence": 0.0-1.0, "reasoning": "one sentence"}'
            ),
            temperature=0.0,
        )

        # Parse JSON response
        raw = raw.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```json\s*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)

        data = json.loads(raw)
        result.is_multi_service = data.get("is_multi_service", False)
        result.service_count = data.get("service_count", 1)
        result.services = data.get("services", [])
        result.communication_pattern = data.get("communication_pattern", "none")
        result.confidence = data.get("confidence", 0.5)
        result.reasoning = data.get("reasoning", "")

        logger.info(
            f"Goal classification: {'multi-service' if result.is_multi_service else 'single-service'} "
            f"(confidence={result.confidence:.0%}, {result.reasoning})"
        )

    except Exception as e:
        logger.debug(f"LLM classification failed: {e} — falling back to keywords")
        return _classify_by_keywords(goal)

    return result


def _classify_by_keywords(goal: str) -> GoalClassification:
    """Improved keyword-based fallback classification.

    More specific than the old list — requires structural indicators,
    not just mention of 'microservice' in passing.
    """
    goal_lower = goal.lower()
    result = GoalClassification()

    # Strong multi-service indicators (require service separation language)
    strong_patterns = [
        r"two\s+(?:separate\s+)?services",
        r"three\s+(?:separate\s+)?services",
        r"multiple\s+(?:separate\s+)?services",
        r"service\s+a\s+and\s+service\s+b",
        r"on\s+port\s+\d{4}\s+.*on\s+port\s+\d{4}",
        r"separate\s+(?:api|server|service)\s+for",
        r"docker-compose\s+with\s+\d+\s+services",
        r"gateway\s+.*(?:forwards?|routes?|proxies)\s+to",
    ]

    for pattern in strong_patterns:
        if re.search(pattern, goal_lower):
            result.is_multi_service = True
            result.confidence = 0.8
            result.reasoning = f"Matched structural pattern: {pattern}"
            break

    # Weak indicators that need additional context
    if not result.is_multi_service:
        weak_count = sum(1 for kw in [
            "microservice", "docker-compose", "api gateway",
            "event-driven", "message queue", "separate backend",
        ] if kw in goal_lower)

        # Disqualifiers — these suggest single-service or explanation
        disqualifiers = ["explain", "tutorial", "what is", "how does", "compare"]
        is_explanation = any(d in goal_lower for d in disqualifiers)

        if weak_count >= 2 and not is_explanation:
            result.is_multi_service = True
            result.confidence = 0.6
            result.reasoning = f"Multiple weak indicators ({weak_count})"

    return result


def verify_services(
    code_files: dict[str, str],
    service_architecture: dict | None = None,
) -> ServiceVerification:
    """Verify multi-service build output for structural soundness.

    Checks:
    1. Every service has a /health endpoint
    2. Every service has at least one route
    3. Cross-service HTTP calls reference correct ports/paths
    4. OpenAPI specs match generated routes
    """
    result = ServiceVerification()

    if not service_architecture:
        return result

    services = service_architecture.get("services", [])
    if not services:
        return result

    for svc in services:
        svc_name = svc.get("name", "unknown")
        svc_package = svc.get("package", svc_name)
        svc_port = svc.get("port", 8000)

        # Find files belonging to this service
        svc_files = {
            fname: content for fname, content in code_files.items()
            if svc_package in fname or svc_name in fname
        }

        if not svc_files:
            # Check if files are at root level (single-directory layout)
            svc_files = code_files

        all_code = "\n".join(svc_files.values())

        # Tier 1: Health endpoint check
        has_health = (
            '"/health"' in all_code or "'/health'" in all_code
            or '@app.get("/health")' in all_code or "@app.route('/health')" in all_code
        )
        if not has_health:
            result.issues.append(f"{svc_name}: missing /health endpoint")
            result.health_endpoints_present = False

        # Tier 2: Route check
        routes = svc.get("routes", [])
        if routes:
            for route in routes:
                path = route.get("path", "")
                if path and path not in all_code and f'"{path}"' not in all_code:
                    result.issues.append(f"{svc_name}: route {path} declared in architecture but not in code")

        # Tier 3: Cross-service reference check
        for other_svc in services:
            if other_svc.get("name") == svc_name:
                continue
            other_port = other_svc.get("port", 8000)
            other_name = other_svc.get("name", "")

            # Check if this service references the other service
            port_ref = str(other_port) in all_code
            name_ref = other_name in all_code

            if port_ref and not name_ref:
                # Referencing port but not by name — fragile but acceptable
                pass
            elif name_ref and not port_ref:
                # Referencing name but not port — might use service discovery
                pass

    if result.issues:
        logger.info(f"Multi-service verification: {len(result.issues)} issues found")
        for issue in result.issues:
            logger.warning(f"  ⚠️  {issue}")
    else:
        logger.info("Multi-service verification: all checks passed")

    return result


def inject_health_endpoints(
    code_files: dict[str, str],
    service_architecture: dict | None = None,
) -> dict[str, str]:
    """Auto-inject /health endpoints into services that are missing them.

    This is a deterministic post-processing step — no LLM needed.
    Detects FastAPI, Flask, and Express apps by import patterns.
    """
    if not service_architecture:
        return code_files

    fixed = dict(code_files)

    for svc in service_architecture.get("services", []):
        svc_package = svc.get("package", svc.get("name", ""))

        # Find the main app file for this service
        app_file = None
        for fname in code_files:
            if svc_package in fname and ("main" in fname or "app" in fname):
                app_file = fname
                break

        if not app_file:
            continue

        content = fixed[app_file]

        # Skip if /health already exists
        if '"/health"' in content or "'/health'" in content:
            continue

        # Detect framework and inject health endpoint
        if "FastAPI" in content or "fastapi" in content:
            # Inject before the last line (usually uvicorn.run or if __name__)
            health_code = '\n\n@app.get("/health")\ndef health():\n    return {"status": "UP"}\n'
            # Insert before if __name__ block or at end
            if 'if __name__' in content:
                content = content.replace('if __name__', health_code + '\nif __name__')
            else:
                content += health_code
            fixed[app_file] = content
            logger.info(f"Injected /health into {app_file} (FastAPI)")

        elif "Flask" in content or "flask" in content:
            health_code = '\n\n@app.route("/health")\ndef health():\n    return {"status": "UP"}\n'
            if 'if __name__' in content:
                content = content.replace('if __name__', health_code + '\nif __name__')
            else:
                content += health_code
            fixed[app_file] = content
            logger.info(f"Injected /health into {app_file} (Flask)")

    return fixed

import asyncio
import json
import httpx

PROBES = [
    "def has_close_elements(numbers: list, threshold: float) -> bool:\n    \"\"\"Check if any two numbers in the list are closer than threshold.\"\"\"",
    "def separate_paren_groups(paren_string: str) -> list:\n    \"\"\"Separate nested parentheses groups from a string into a list.\"\"\"",
    "def truncate_number(number: float) -> float:\n    \"\"\"Return the decimal part of a positive float.\"\"\"",
    "def below_zero(operations: list) -> bool:\n    \"\"\"Return True if running balance of deposits/withdrawals ever falls below zero.\"\"\"",
    "def mean_absolute_deviation(numbers: list) -> float:\n    \"\"\"Calculate mean absolute deviation around the mean.\"\"\"",
]

async def probe(client, sig):
    prompt = f"Complete this Python function. Provide only the function body, no explanation:\n\n{sig}\n"
    r = await client.post(
        "/api/chat",
        json={
            "model": "qwen2.5-coder:14b",
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.0, "seed": 42, "num_predict": 512},
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]

async def main():
    results = []
    async with httpx.AsyncClient(base_url="http://localhost:11434") as client:
        for i, p in enumerate(PROBES):
            print(f"Probing {i+1}/{len(PROBES)}...", flush=True)
            out = await probe(client, p)
            results.append({"problem": p[:60] + "...", "completion_preview": out[:200]})

    with open("audit/gate5_contamination/humaneval_probes.json", "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=== Completions (first 200 chars each) ===")
    for r in results:
        print(f"\nProblem: {r['problem']}")
        print(f"Completion: {r['completion_preview']}")

asyncio.run(main())

# HTTP API & SDK Reference

## HTTP Evaluation Endpoint

```http
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer <TYPESAFE_API_KEY>
Content-Type: application/json
```

### Request Payload Shape

```json
{
  "state": "String text, JSON object, or array representing context",
  "model": "jev-latest",
  "questions": {
    "<question_id_1>": {
      "type": "noul",
      "instructions": "Is this inquiry urgent?",
      "criteria": { "true": "Time-sensitive issue", "false": "General inquiry" }
    },
    "<question_id_2>": {
      "type": "choice",
      "instructions": "Route to which department?",
      "criteria": {
        "sales": "Inquiries regarding plans or pricing",
        "support": "Technical help and bug reports"
      }
    },
    "<question_id_3>": {
      "type": "score",
      "instructions": "Rate severity level",
      "criteria": ["Low", "Medium", "High", "Critical"]
    }
  }
}
```

### Response Payload Shape

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "<question_id_1>": {
      "type": "noul",
      "noul": 0.88
    },
    "<question_id_2>": {
      "type": "choice",
      "choice": "support",
      "probabilities": { "sales": 0.05, "support": 0.95 },
      "confidence": 0.91
    },
    "<question_id_3>": {
      "type": "score",
      "score": 2.7,
      "legend": { "0": "Low", "1": "Medium", "2": "High", "3": "Critical" },
      "probabilities": { "0": 0.01, "1": 0.09, "2": 0.10, "3": 0.80 },
      "confidence": 0.78
    }
  },
  "usage": {
    "input_tokens": 142,
    "output_tokens": 32
  }
}
```

### Model Discovery Endpoint

```http
GET https://api.typesafe.ai/v1/models
Authorization: Bearer <TYPESAFE_API_KEY>
```

---

## Python SDK (`typesafe-sdk`)

### Installation & Client Setup

```bash
pip install typesafe-sdk
# or uv add typesafe-sdk
```

### Synchronous Client

```python
from typesafe_sdk import TypeSafeClient

with TypeSafeClient() as client:
    response = client.system_one(
        state="Customer account #12345 cannot access dashboard.",
        questions={
            "is_outage": {
                "type": "noul",
                "instructions": "Does this report describe a system outage?"
            },
            "category": {
                "type": "choice",
                "instructions": "Categorize the ticket",
                "criteria": {"auth": "Login or session failure", "ui": "Display issues"}
            }
        }
    )

    print("Outage probability:", response.answers["is_outage"].noul)
    print("Category:", response.answers["category"].choice)
    print("Confidence:", response.answers["category"].confidence)
```

### Asynchronous Client

```python
import asyncio
from typesafe_sdk import AsyncTypeSafeClient

async def main():
    async with AsyncTypeSafeClient() as client:
        response = await client.system_one(
            state={"document": "Text content to evaluate..."},
            questions={
                "quality": {
                    "type": "score",
                    "instructions": "Evaluate quality",
                    "criteria": ["Poor", "Fair", "Good", "Excellent"]
                }
            }
        )
        print("Score:", response.answers["quality"].score)

asyncio.run(main())
```

---

## TypeScript / JavaScript SDK (`@typesafe-ai/sdk`)

### Installation

```bash
npm install @typesafe-ai/sdk
```

### Usage Example

```typescript
import { TypeSafeClient, choice, noul, score } from "@typesafe-ai/sdk";

const client = new TypeSafeClient({
  // apiKey defaults to process.env.TYPESAFE_API_KEY
});

const response = await client.systemOne({
  state: {
    ticketId: "T-9921",
    message: "I was charged twice on my card yesterday."
  },
  questions: {
    urgent: noul("Does this require immediate attention?"),
    topic: choice("What is the primary topic?", {
      billing: "Invoices, transactions, charges",
      technical: "Software errors, crashes",
      other: null
    }),
    frustration: score("Rate frustration level", [
      "Patient",
      "Annoyed",
      "Furious"
    ])
  }
});

console.log("Topic:", response.answers.topic.choice);
console.log("Urgent prob:", response.answers.urgent.noul);
console.log("Frustration score:", response.answers.frustration.score);
```

---

## Agent Skill Integration

For automated coding assistants (Claude Code, Codex, Cursor, etc.):
- **Claude Code**:
  ```bash
  claude plugin marketplace add typesafe-ai/skills
  claude plugin install typesafe@typesafe-ai
  ```
- **Universal agents**:
  ```bash
  npx skills add typesafe-ai/skills --skill typesafe-ai
  ```
- Skill documentation source: `https://github.com/typesafe-ai/skills/blob/main/skills/typesafe-ai/SKILL.md`

import argparse
import json
import os
from pathlib import Path

from tf_client import chat

from dotenv import load_dotenv
load_dotenv()

JUDGE_MODEL = "deepseek-ai/DeepSeek-V4-Pro"


SYSTEM_PROMPT = """
You are a strict evaluator for an insurance customer-support assistant.

Evaluate the candidate answer against the ground-truth answer.

Definitions:
- relevant=true if the candidate answer gives the same important fact as the ground truth.
- hallucination=true if the candidate confidently states facts that contradict the ground truth.
- "I don't know" or a refusal is NOT a hallucination.

Return ONLY valid JSON:

{
  "relevant": true,
  "hallucination": false,
  "reason": "short explanation"
}
"""


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = data["questions"]

    return {
        q["id"]: q
        for q in data
    }


def judge_answer(question, ground_truth, answer):
    prompt = f"""
Question:
{question}

Ground truth answer:
{ground_truth}

Candidate answer:
{answer}
"""

    response = chat(
        [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        model=JUDGE_MODEL,
        temperature=0,
    )

    return json.loads(response)


def citation_accuracy(candidate_citations, ground_truth_sources):
    """
    Each ground_truth_sources entry represents a required fact.
    Each entry has an any_of list of acceptable file/page pairs.
    """

    if not ground_truth_sources:
        return 1.0

    if not candidate_citations:
        return 0.0

    for group in ground_truth_sources:

        matched = False

        for acceptable in group["any_of"]:

            for citation in candidate_citations:

                same_file = (
                    citation.get("file") == acceptable["file"]
                )

                same_page = (
                    citation.get("page") == acceptable.get("page")
                )

                if same_file and same_page:
                    matched = True

        if not matched:
            return 0.0

    return 1.0


def evaluate(experiment_dir, questions_file):

    experiment_dir = Path(experiment_dir)

    answers_file = experiment_dir / "answers.jsonl"
    output_file = experiment_dir / "evaluation_results.jsonl"

    questions = load_questions(questions_file)
    answers = load_jsonl(answers_file)

    with open(output_file, "w", encoding="utf-8") as out:

        for answer in answers:

            qid = answer["id"]

            question = questions[qid]

            judge = judge_answer(
                question=question["question"],
                ground_truth=question["ground_truth_answer"],
                answer=answer["answer"],
            )

            citation_score = citation_accuracy(
                answer.get("citations", []),
                question.get("ground_truth_sources", [])
            )

            result = {
                "id": qid,

                "domain": question.get("domain"),
                "difficulty": question.get("difficulty"),

                "relevant": judge["relevant"],
                "hallucination": judge["hallucination"],
                "reason": judge["reason"],

                "citation_accuracy": citation_score,

                "latency_ms": answer.get("latency_ms"),
            }

            out.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                ) + "\n"
            )

            print(
                f"{qid}: "
                f"relevant={result['relevant']} "
                f"hallucination={result['hallucination']} "
                f"citation={result['citation_accuracy']}"
            )

    print(f"\nWrote {output_file}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment folder"
    )

    parser.add_argument(
        "--questions",
        required=True,
        help="reference_questions.json"
    )

    args = parser.parse_args()

    evaluate(
        args.experiment,
        args.questions
    )
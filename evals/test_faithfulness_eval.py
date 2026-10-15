import os
import time
import pytest
import dotenv
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from langchain_core.messages import HumanMessage
from app.langgraph.nodes import llm, get_system_prompt
from app.api.chat import extract_text_content

dotenv.load_dotenv()

# Benchmark evaluation dataset: (Business Context, Customer Query, Description)
BENCHMARK_CASES = [
    {
        "id": "pizzeria_hours_and_menu",
        "business_context": (
            "Business Name: Bella Napoli Pizzeria. "
            "Services: Handcrafted Neapolitan Pizza, Calzones, Tiramisu, Express Delivery. "
            "Timings: 11:30 AM to 10:30 PM Tuesday through Sunday. Closed on Mondays."
        ),
        "query": "What are your opening hours and what food items do you serve?",
        "expected_topics": ["11:30 AM", "10:30 PM", "Pizza", "Calzones"],
    },
    {
        "id": "unsupported_service_hallucination_check",
        "business_context": (
            "Business Name: Bella Napoli Pizzeria. "
            "Services: Handcrafted Neapolitan Pizza, Calzones, Tiramisu, Express Delivery. "
            "Timings: 11:30 AM to 10:30 PM Tuesday through Sunday. Closed on Mondays."
        ),
        "query": "Can you do an oil change for my car while I wait?",
        "expected_topics": ["do not offer", "cannot", "pizza"],
    },
    {
        "id": "healthcare_timings_constraint",
        "business_context": (
            "Business Name: Apex Dental Clinic. "
            "Services: Dental Implants, Teeth Whitening, Routine Cleanings, Emergency Consultations. "
            "Timings: Monday to Friday 9:00 AM to 5:00 PM. Closed on Saturdays and Sundays."
        ),
        "query": "Can I book a teeth cleaning appointment this Sunday at 2 PM?",
        "expected_topics": ["closed", "Sunday", "Monday to Friday"],
    },
]


@pytest.mark.eval
@pytest.mark.parametrize("case", BENCHMARK_CASES, ids=[c["id"] for c in BENCHMARK_CASES])
def test_business_query_faithfulness(gemini_judge, case):
    """
    Evaluates whether the LLM's response to business inquiries is strictly faithful
    to the injected business context without hallucinating unsupported claims or facts.
    """
    # 1. Generate live response through Keystone's actual query system prompt & model
    sys_msg, version = get_system_prompt(
        "query",
        state={},
        business_context=case["business_context"]
    )
    response = llm.invoke([sys_msg, HumanMessage(content=case["query"])])
    actual_output = extract_text_content(response)

    print(f"\n[EVAL ID]: {case['id']}")
    print(f"[QUERY]: {case['query']}")
    print(f"[ACTUAL OUTPUT]:\n{actual_output}\n")

    # 2. Build DeepEval test case
    test_case = LLMTestCase(
        input=case["query"],
        actual_output=actual_output,
        retrieval_context=[case["business_context"]],
    )

    # 3. Measure Faithfulness Metric (Claims must be supported by retrieval_context)
    faithfulness_metric = FaithfulnessMetric(
        threshold=0.75,
        model=gemini_judge,
        include_reason=True
    )
    faithfulness_metric.measure(test_case)

    print(f"[FAITHFULNESS SCORE]: {faithfulness_metric.score}")
    print(f"[FAITHFULNESS REASON]: {faithfulness_metric.reason}")

    assert faithfulness_metric.is_successful(), (
        f"Faithfulness eval failed for '{case['id']}': score={faithfulness_metric.score}, "
        f"reason={faithfulness_metric.reason}"
    )
    # Small pause to respect Gemini rate limits
    time.sleep(2)

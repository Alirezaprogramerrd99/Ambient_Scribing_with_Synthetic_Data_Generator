"""Quick test of OpenAI integration"""
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

# Verify key is set
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY not set!")
    print("Create a .env file with: OPENAI_API_KEY=sk-your-key-here")
    exit(1)

print(f"✓ API key found: {api_key[:8]}...{api_key[-4:]}")

# Test the teacher directly
from src.teacher import create_teacher

teacher = create_teacher(
    provider="openai",
    model_name="gpt-4o-mini",
    temperature=0.7,
    use_rag=True,  # Skip RAG for this test
)

print(f"✓ Teacher created: {teacher.model_name}")

# Test a simple generation
scenario = "45-year-old male with chest pain for 2 hours, radiating to left arm"

print(f"\nGenerating sample for: {scenario}")
print("This should take ~10-30 seconds with OpenAI...\n")

result = teacher.generate(scenario)

if result.success:
    print("✓ Generation successful!")
    print(f"  Dialogue turns: {len(result.sample.dialogue)}")
    print(f"  Chief complaint: {result.sample.summary.chief_complaint}")
    print(f"  Generation time: {result.generation_time_seconds:.1f}s")
else:
    print(f"❌ Generation failed: {result.error}")